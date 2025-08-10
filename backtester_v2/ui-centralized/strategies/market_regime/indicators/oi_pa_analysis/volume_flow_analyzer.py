"""
Volume Flow Analyzer - Institutional vs Retail Detection
=======================================================

Analyzes volume flows to detect institutional vs retail activity patterns
and calculate flow-based sentiment indicators.

Author: Market Regime Refactoring Team  
Date: 2025-07-06
Version: 2.0.0 - Enhanced Volume Flow Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FlowAnalysisResult:
    """Result structure for flow analysis"""
    institutional_flow: Dict[str, float]
    retail_flow: Dict[str, float]
    flow_sentiment: float
    institutional_ratio: float
    flow_divergence: float
    flow_quality: float

class VolumeFlowAnalyzer:
    """
    Advanced volume flow analysis for institutional detection
    
    Features:
    - Institutional vs retail classification by size
    - Flow sentiment calculation
    - Session-based flow analysis
    - Flow persistence tracking
    - Multi-dimensional flow metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Volume Flow Analyzer"""
        self.config = config or {}
        
        # Institutional detection thresholds
        self.institutional_oi_threshold = self.config.get('institutional_oi_threshold', 10000)
        self.institutional_volume_threshold = self.config.get('institutional_volume_threshold', 5000)
        self.large_block_threshold = self.config.get('large_block_threshold', 20000)
        
        # Flow analysis parameters
        self.flow_significance_threshold = self.config.get('flow_significance_threshold', 1000)
        self.sentiment_calculation_method = self.config.get('sentiment_calculation_method', 'weighted')
        
        # Session analysis
        self.enable_session_analysis = self.config.get('enable_session_analysis', True)
        self.session_weights = self.config.get('session_weights', {
            'morning': 1.2,    # 9:15-11:00
            'midday': 1.0,     # 11:00-14:00  
            'evening': 1.3     # 14:00-15:30
        })
        
        # Flow tracking
        self.flow_history = []
        self.institutional_patterns = {}
        
        logger.info("VolumeFlowAnalyzer initialized for institutional detection")
    
    def analyze_volume_flows(self, 
                           market_data: pd.DataFrame,
                           selected_strikes: List,
                           current_time: Optional[datetime] = None) -> FlowAnalysisResult:
        """
        Analyze volume flows for institutional vs retail detection
        
        Args:
            market_data: Market data DataFrame
            selected_strikes: Selected strikes for analysis
            current_time: Current timestamp for session analysis
            
        Returns:
            FlowAnalysisResult: Comprehensive flow analysis
        """
        try:
            current_time = current_time or datetime.now()
            
            # Initialize flow containers
            institutional_flow = {'calls': 0, 'puts': 0, 'total': 0}
            retail_flow = {'calls': 0, 'puts': 0, 'total': 0}
            
            flow_details = []
            
            # Analyze each selected strike
            for strike_info in selected_strikes:
                strike = strike_info.strike
                option_type = strike_info.option_type
                
                strike_data = market_data[
                    (market_data['strike'] == strike) & 
                    (market_data['option_type'] == option_type)
                ]
                
                if strike_data.empty:
                    continue
                
                row = strike_data.iloc[0]
                
                # Extract OI and Volume data
                if option_type == 'CE':
                    oi = row.get('ce_oi', 0)
                    volume = row.get('ce_volume', 0)
                    flow_type = 'calls'
                else:  # PE
                    oi = row.get('pe_oi', 0)
                    volume = row.get('pe_volume', 0)
                    flow_type = 'puts'
                
                # Calculate total flow for this strike
                total_flow = oi + volume
                
                if total_flow < self.flow_significance_threshold:
                    continue
                
                # Classify as institutional or retail
                is_institutional = self._classify_institutional_flow(oi, volume, total_flow)
                
                flow_details.append({
                    'strike': strike,
                    'option_type': option_type,
                    'oi': oi,
                    'volume': volume,
                    'total_flow': total_flow,
                    'is_institutional': is_institutional,
                    'flow_type': flow_type
                })
                
                # Aggregate flows
                if is_institutional:
                    institutional_flow[flow_type] += total_flow
                    institutional_flow['total'] += total_flow
                else:
                    retail_flow[flow_type] += total_flow
                    retail_flow['total'] += total_flow
            
            # Apply session weighting if enabled
            if self.enable_session_analysis:
                institutional_flow, retail_flow = self._apply_session_weighting(
                    institutional_flow, retail_flow, current_time
                )
            
            # Calculate flow metrics
            flow_sentiment = self._calculate_flow_sentiment(institutional_flow, retail_flow)
            institutional_ratio = self._calculate_institutional_ratio(institutional_flow, retail_flow)
            flow_divergence = self._calculate_flow_divergence(institutional_flow, retail_flow)
            flow_quality = self._assess_flow_quality(flow_details)
            
            # Record flow analysis
            self._record_flow_analysis(institutional_flow, retail_flow, flow_sentiment)
            
            return FlowAnalysisResult(
                institutional_flow=institutional_flow,
                retail_flow=retail_flow,
                flow_sentiment=flow_sentiment,
                institutional_ratio=institutional_ratio,
                flow_divergence=flow_divergence,
                flow_quality=flow_quality
            )
            
        except Exception as e:
            logger.error(f"Error analyzing volume flows: {e}")
            return self._get_default_flow_result()
    
    def _classify_institutional_flow(self, oi: float, volume: float, total_flow: float) -> bool:
        """Classify flow as institutional or retail"""
        try:
            # Multiple criteria for institutional classification
            criteria_met = 0
            
            # Criterion 1: Large OI
            if oi >= self.institutional_oi_threshold:
                criteria_met += 1
            
            # Criterion 2: Large Volume
            if volume >= self.institutional_volume_threshold:
                criteria_met += 1
            
            # Criterion 3: Large total flow
            if total_flow >= self.large_block_threshold:
                criteria_met += 1
            
            # Criterion 4: OI-Volume ratio (institutional prefer OI)
            if oi > 0 and volume > 0:
                oi_ratio = oi / (oi + volume)
                if oi_ratio > 0.6:  # 60% or more is OI
                    criteria_met += 1
            
            # Classify as institutional if 2 or more criteria met
            return criteria_met >= 2
            
        except Exception as e:
            logger.error(f"Error classifying institutional flow: {e}")
            return False
    
    def _apply_session_weighting(self, 
                               institutional_flow: Dict[str, float],
                               retail_flow: Dict[str, float],
                               current_time: datetime) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Apply session-based weighting to flows"""
        try:
            hour = current_time.hour
            minute = current_time.minute
            
            # Determine session
            if 9 <= hour < 11:
                session = 'morning'
            elif 11 <= hour < 14:
                session = 'midday'
            elif 14 <= hour < 16:
                session = 'evening'
            else:
                session = 'midday'  # Default
            
            weight = self.session_weights.get(session, 1.0)
            
            # Apply weight to institutional flows (they're more session-sensitive)
            weighted_institutional = {
                key: value * weight for key, value in institutional_flow.items()
            }
            
            # Retail flows get inverse weighting (less session-sensitive)
            retail_weight = 2.0 - weight  # Inverse relationship
            weighted_retail = {
                key: value * retail_weight for key, value in retail_flow.items()
            }
            
            return weighted_institutional, weighted_retail
            
        except Exception as e:
            logger.error(f"Error applying session weighting: {e}")
            return institutional_flow, retail_flow
    
    def _calculate_flow_sentiment(self, 
                                institutional_flow: Dict[str, float],
                                retail_flow: Dict[str, float]) -> float:
        """Calculate flow-based sentiment"""
        try:
            if self.sentiment_calculation_method == 'weighted':
                return self._calculate_weighted_flow_sentiment(institutional_flow, retail_flow)
            else:
                return self._calculate_simple_flow_sentiment(institutional_flow, retail_flow)
                
        except Exception as e:
            logger.error(f"Error calculating flow sentiment: {e}")
            return 0.0
    
    def _calculate_weighted_flow_sentiment(self, 
                                         institutional_flow: Dict[str, float],
                                         retail_flow: Dict[str, float]) -> float:
        """Calculate weighted flow sentiment (institutional weighted higher)"""
        try:
            # Institutional sentiment (weighted 70%)
            inst_total = institutional_flow['total']
            if inst_total > 0:
                inst_sentiment = (institutional_flow['calls'] - institutional_flow['puts']) / inst_total
            else:
                inst_sentiment = 0.0
            
            # Retail sentiment (weighted 30%)
            retail_total = retail_flow['total']
            if retail_total > 0:
                retail_sentiment = (retail_flow['calls'] - retail_flow['puts']) / retail_total
            else:
                retail_sentiment = 0.0
            
            # Weighted combination
            institutional_weight = 0.7
            retail_weight = 0.3
            
            weighted_sentiment = (institutional_weight * inst_sentiment + 
                                retail_weight * retail_sentiment)
            
            return np.clip(weighted_sentiment, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating weighted flow sentiment: {e}")
            return 0.0
    
    def _calculate_simple_flow_sentiment(self, 
                                       institutional_flow: Dict[str, float],
                                       retail_flow: Dict[str, float]) -> float:
        """Calculate simple combined flow sentiment"""
        try:
            total_calls = institutional_flow['calls'] + retail_flow['calls']
            total_puts = institutional_flow['puts'] + retail_flow['puts']
            total_flow = total_calls + total_puts
            
            if total_flow > 0:
                sentiment = (total_calls - total_puts) / total_flow
                return np.clip(sentiment, -1.0, 1.0)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating simple flow sentiment: {e}")
            return 0.0
    
    def _calculate_institutional_ratio(self, 
                                     institutional_flow: Dict[str, float],
                                     retail_flow: Dict[str, float]) -> float:
        """Calculate institutional participation ratio"""
        try:
            total_flow = institutional_flow['total'] + retail_flow['total']
            if total_flow > 0:
                return institutional_flow['total'] / total_flow
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_flow_divergence(self, 
                                 institutional_flow: Dict[str, float],
                                 retail_flow: Dict[str, float]) -> float:
        """Calculate divergence between institutional and retail sentiment"""
        try:
            # Calculate individual sentiments
            inst_total = institutional_flow['total']
            retail_total = retail_flow['total']
            
            if inst_total > 0:
                inst_sentiment = (institutional_flow['calls'] - institutional_flow['puts']) / inst_total
            else:
                inst_sentiment = 0.0
            
            if retail_total > 0:
                retail_sentiment = (retail_flow['calls'] - retail_flow['puts']) / retail_total
            else:
                retail_sentiment = 0.0
            
            # Divergence is absolute difference
            return abs(inst_sentiment - retail_sentiment)
            
        except Exception as e:
            logger.error(f"Error calculating flow divergence: {e}")
            return 0.0
    
    def _assess_flow_quality(self, flow_details: List[Dict]) -> float:
        """Assess quality of flow analysis"""
        try:
            if not flow_details:
                return 0.0
            
            quality_factors = []
            
            # Factor 1: Number of strikes with significant flow
            significant_flows = len([fd for fd in flow_details if fd['total_flow'] >= self.flow_significance_threshold])
            strike_coverage = min(significant_flows / 10, 1.0)  # Normalize to max 10 strikes
            quality_factors.append(strike_coverage)
            
            # Factor 2: Flow distribution (not concentrated in one strike)
            total_flows = [fd['total_flow'] for fd in flow_details]
            if len(total_flows) > 1:
                flow_std = np.std(total_flows)
                flow_mean = np.mean(total_flows)
                if flow_mean > 0:
                    flow_distribution = 1.0 - min(flow_std / flow_mean, 1.0)
                    quality_factors.append(flow_distribution)
            
            # Factor 3: Institutional vs retail balance
            institutional_count = len([fd for fd in flow_details if fd['is_institutional']])
            retail_count = len(flow_details) - institutional_count
            
            if len(flow_details) > 0:
                balance_ratio = min(institutional_count, retail_count) / len(flow_details)
                quality_factors.append(balance_ratio * 2)  # Scale to make 50-50 = 1.0
            
            # Average quality factors
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            logger.error(f"Error assessing flow quality: {e}")
            return 0.5
    
    def _record_flow_analysis(self, 
                            institutional_flow: Dict[str, float],
                            retail_flow: Dict[str, float],
                            flow_sentiment: float):
        """Record flow analysis for tracking"""
        try:
            record = {
                'timestamp': datetime.now(),
                'institutional_total': institutional_flow['total'],
                'retail_total': retail_flow['total'],
                'institutional_ratio': institutional_flow['total'] / (institutional_flow['total'] + retail_flow['total']) if (institutional_flow['total'] + retail_flow['total']) > 0 else 0,
                'flow_sentiment': flow_sentiment
            }
            
            self.flow_history.append(record)
            
            # Keep only last 100 records
            if len(self.flow_history) > 100:
                self.flow_history = self.flow_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording flow analysis: {e}")
    
    def analyze_institutional_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in institutional activity"""
        try:
            if len(self.flow_history) < 5:
                return {'status': 'insufficient_data'}
            
            recent_history = self.flow_history[-20:]
            
            # Institutional participation trend
            inst_ratios = [h['institutional_ratio'] for h in recent_history]
            avg_participation = np.mean(inst_ratios)
            participation_trend = 'increasing' if inst_ratios[-1] > inst_ratios[0] else 'decreasing'
            
            # Sentiment consistency
            sentiments = [h['flow_sentiment'] for h in recent_history]
            sentiment_consistency = 1.0 - np.std(sentiments)  # Lower std = higher consistency
            
            # Activity level assessment
            inst_totals = [h['institutional_total'] for h in recent_history]
            avg_activity = np.mean(inst_totals)
            activity_level = 'high' if avg_activity > 50000 else 'medium' if avg_activity > 20000 else 'low'
            
            return {
                'average_participation': avg_participation,
                'participation_trend': participation_trend,
                'sentiment_consistency': sentiment_consistency,
                'activity_level': activity_level,
                'current_sentiment': sentiments[-1] if sentiments else 0.0,
                'analysis_period': len(recent_history)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing institutional patterns: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_default_flow_result(self) -> FlowAnalysisResult:
        """Get default flow result for error cases"""
        return FlowAnalysisResult(
            institutional_flow={'calls': 0, 'puts': 0, 'total': 0},
            retail_flow={'calls': 0, 'puts': 0, 'total': 0},
            flow_sentiment=0.0,
            institutional_ratio=0.0,
            flow_divergence=0.0,
            flow_quality=0.0
        )
    
    def get_flow_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of flow analysis performance"""
        try:
            summary = {
                'total_analyses': len(self.flow_history),
                'classification_thresholds': {
                    'institutional_oi_threshold': self.institutional_oi_threshold,
                    'institutional_volume_threshold': self.institutional_volume_threshold,
                    'large_block_threshold': self.large_block_threshold
                },
                'session_analysis_enabled': self.enable_session_analysis,
                'sentiment_method': self.sentiment_calculation_method
            }
            
            if self.flow_history:
                recent_history = self.flow_history[-10:]
                summary['recent_performance'] = {
                    'avg_institutional_ratio': np.mean([h['institutional_ratio'] for h in recent_history]),
                    'avg_flow_sentiment': np.mean([h['flow_sentiment'] for h in recent_history]),
                    'sentiment_volatility': np.std([h['flow_sentiment'] for h in recent_history])
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating flow analysis summary: {e}")
            return {'status': 'error', 'error': str(e)}