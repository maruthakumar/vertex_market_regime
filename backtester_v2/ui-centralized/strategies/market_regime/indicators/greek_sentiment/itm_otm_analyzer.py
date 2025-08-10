"""
ITM/OTM Analyzer - Institutional Sentiment Detection
===================================================

Analyzes In-The-Money and Out-of-The-Money options to detect institutional
sentiment and trading patterns. This is a key enhancement for understanding
institutional vs retail activity in the options market.

Author: Market Regime Refactoring Team
Date: 2025-07-06
Version: 2.0.0 - ITM/OTM Analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ITMOTMAnalyzer:
    """
    ITM/OTM options analyzer for institutional sentiment detection
    
    Key Features:
    - ITM analysis for institutional sentiment
    - OTM analysis for speculative activity  
    - Moneyness-based classification
    - Volume and OI flow analysis
    - Time-based pattern detection
    - Risk-adjusted sentiment scoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ITM/OTM Analyzer"""
        self.config = config or {}
        
        # Moneyness thresholds
        self.itm_threshold = self.config.get('itm_threshold', 0.02)  # 2% ITM threshold
        self.deep_itm_threshold = self.config.get('deep_itm_threshold', 0.05)  # 5% deep ITM
        self.deep_otm_threshold = self.config.get('deep_otm_threshold', 0.05)  # 5% deep OTM
        
        # Analysis weights
        self.oi_weight = self.config.get('oi_weight', 0.6)
        self.volume_weight = self.config.get('volume_weight', 0.4)
        
        # Institutional detection parameters
        self.institutional_size_threshold = self.config.get('institutional_size_threshold', 10000)
        self.bulk_activity_threshold = self.config.get('bulk_activity_threshold', 0.3)  # 30% of total
        
        # Time-based analysis
        self.enable_time_analysis = self.config.get('enable_time_analysis', True)
        
        # Sentiment contribution limits
        self.max_itm_contribution = self.config.get('max_itm_contribution', 0.3)  # 30% max
        self.max_otm_contribution = self.config.get('max_otm_contribution', 0.2)  # 20% max
        
        # Analysis history
        self.analysis_history = []
        
        logger.info(f"ITMOTMAnalyzer initialized: ITM threshold={self.itm_threshold:.1%}")
    
    def analyze_itm_otm_sentiment(self, 
                                market_data: pd.DataFrame,
                                spot_price: float,
                                current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze ITM/OTM sentiment for institutional detection
        
        Args:
            market_data: Option market data
            spot_price: Current underlying spot price
            current_time: Current timestamp for time-based analysis
            
        Returns:
            Dict: Comprehensive ITM/OTM analysis results
        """
        try:
            current_time = current_time or datetime.now()
            
            # Classify options by moneyness
            moneyness_classification = self._classify_options_by_moneyness(market_data, spot_price)
            
            # Analyze ITM sentiment (institutional)
            itm_analysis = self._analyze_itm_sentiment(
                market_data, moneyness_classification, spot_price
            )
            
            # Analyze OTM sentiment (speculative)
            otm_analysis = self._analyze_otm_sentiment(
                market_data, moneyness_classification, spot_price
            )
            
            # Analyze flow patterns
            flow_analysis = self._analyze_option_flows(
                market_data, moneyness_classification, spot_price
            )
            
            # Time-based analysis if enabled
            time_analysis = {}
            if self.enable_time_analysis:
                time_analysis = self._analyze_time_patterns(
                    market_data, moneyness_classification, current_time
                )
            
            # Calculate combined sentiment scores
            sentiment_scores = self._calculate_sentiment_scores(
                itm_analysis, otm_analysis, flow_analysis
            )
            
            # Risk-adjusted scoring
            risk_adjusted_scores = self._apply_risk_adjustments(
                sentiment_scores, moneyness_classification
            )
            
            # Compile comprehensive results
            analysis_result = {
                'itm_analysis': itm_analysis,
                'otm_analysis': otm_analysis,
                'flow_analysis': flow_analysis,
                'time_analysis': time_analysis,
                'sentiment_scores': sentiment_scores,
                'risk_adjusted_scores': risk_adjusted_scores,
                'moneyness_distribution': self._get_moneyness_distribution(moneyness_classification),
                'institutional_indicators': self._detect_institutional_activity(itm_analysis, otm_analysis),
                'timestamp': current_time,
                'spot_price': spot_price
            }
            
            # Store in history
            self._record_analysis(analysis_result)
            
            logger.debug(f"ITM/OTM analysis completed: ITM sentiment={sentiment_scores.get('itm_sentiment', 0):.3f}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in ITM/OTM sentiment analysis: {e}")
            return self._get_default_analysis()
    
    def _classify_options_by_moneyness(self, 
                                     market_data: pd.DataFrame,
                                     spot_price: float) -> Dict[str, pd.DataFrame]:
        """Classify options by moneyness relative to spot price"""
        try:
            classification = {
                'deep_itm_calls': pd.DataFrame(),
                'itm_calls': pd.DataFrame(),
                'atm_calls': pd.DataFrame(),
                'otm_calls': pd.DataFrame(),
                'deep_otm_calls': pd.DataFrame(),
                'deep_itm_puts': pd.DataFrame(),
                'itm_puts': pd.DataFrame(),
                'atm_puts': pd.DataFrame(),
                'otm_puts': pd.DataFrame(),
                'deep_otm_puts': pd.DataFrame()
            }
            
            # Separate calls and puts
            calls = market_data[market_data['option_type'] == 'CE'].copy()
            puts = market_data[market_data['option_type'] == 'PE'].copy()
            
            if not calls.empty:
                calls['moneyness'] = (calls['strike'] - spot_price) / spot_price
                
                # Classify calls
                classification['deep_itm_calls'] = calls[calls['moneyness'] <= -self.deep_itm_threshold]
                classification['itm_calls'] = calls[
                    (calls['moneyness'] > -self.deep_itm_threshold) & 
                    (calls['moneyness'] <= -self.itm_threshold)
                ]
                classification['atm_calls'] = calls[
                    (calls['moneyness'] > -self.itm_threshold) & 
                    (calls['moneyness'] <= self.itm_threshold)
                ]
                classification['otm_calls'] = calls[
                    (calls['moneyness'] > self.itm_threshold) & 
                    (calls['moneyness'] <= self.deep_otm_threshold)
                ]
                classification['deep_otm_calls'] = calls[calls['moneyness'] > self.deep_otm_threshold]
            
            if not puts.empty:
                puts['moneyness'] = (spot_price - puts['strike']) / spot_price
                
                # Classify puts
                classification['deep_itm_puts'] = puts[puts['moneyness'] <= -self.deep_itm_threshold]
                classification['itm_puts'] = puts[
                    (puts['moneyness'] > -self.deep_itm_threshold) & 
                    (puts['moneyness'] <= -self.itm_threshold)
                ]
                classification['atm_puts'] = puts[
                    (puts['moneyness'] > -self.itm_threshold) & 
                    (puts['moneyness'] <= self.itm_threshold)
                ]
                classification['otm_puts'] = puts[
                    (puts['moneyness'] > self.itm_threshold) & 
                    (puts['moneyness'] <= self.deep_otm_threshold)
                ]
                classification['deep_otm_puts'] = puts[puts['moneyness'] > self.deep_otm_threshold]
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying options by moneyness: {e}")
            return {}
    
    def _analyze_itm_sentiment(self, 
                             market_data: pd.DataFrame,
                             moneyness_classification: Dict[str, pd.DataFrame],
                             spot_price: float) -> Dict[str, Any]:
        """Analyze ITM options for institutional sentiment"""
        try:
            itm_calls = pd.concat([
                moneyness_classification.get('deep_itm_calls', pd.DataFrame()),
                moneyness_classification.get('itm_calls', pd.DataFrame())
            ])
            
            itm_puts = pd.concat([
                moneyness_classification.get('deep_itm_puts', pd.DataFrame()),
                moneyness_classification.get('itm_puts', pd.DataFrame())
            ])
            
            # Analyze ITM call activity
            call_analysis = self._analyze_option_group(itm_calls, 'ce', 'ITM_CALLS')
            
            # Analyze ITM put activity  
            put_analysis = self._analyze_option_group(itm_puts, 'pe', 'ITM_PUTS')
            
            # Calculate net ITM sentiment
            total_call_flow = call_analysis['total_weighted_flow']
            total_put_flow = put_analysis['total_weighted_flow']
            total_flow = total_call_flow + total_put_flow
            
            if total_flow > 0:
                net_itm_sentiment = (total_call_flow - total_put_flow) / total_flow
            else:
                net_itm_sentiment = 0.0
            
            # Institutional activity indicators
            institutional_indicators = {
                'large_block_activity': self._detect_large_blocks(itm_calls, itm_puts),
                'unusual_oi_buildup': self._detect_unusual_oi_buildup(itm_calls, itm_puts),
                'cross_strike_coordination': self._detect_cross_strike_patterns(itm_calls, itm_puts)
            }
            
            return {
                'call_analysis': call_analysis,
                'put_analysis': put_analysis,
                'net_sentiment': net_itm_sentiment,
                'total_activity': total_flow,
                'institutional_indicators': institutional_indicators,
                'bullish_bias': total_call_flow / total_flow if total_flow > 0 else 0.5,
                'bearish_bias': total_put_flow / total_flow if total_flow > 0 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ITM sentiment: {e}")
            return {}
    
    def _analyze_otm_sentiment(self,
                             market_data: pd.DataFrame,
                             moneyness_classification: Dict[str, pd.DataFrame],
                             spot_price: float) -> Dict[str, Any]:
        """Analyze OTM options for speculative sentiment"""
        try:
            otm_calls = pd.concat([
                moneyness_classification.get('otm_calls', pd.DataFrame()),
                moneyness_classification.get('deep_otm_calls', pd.DataFrame())
            ])
            
            otm_puts = pd.concat([
                moneyness_classification.get('otm_puts', pd.DataFrame()),
                moneyness_classification.get('deep_otm_puts', pd.DataFrame())
            ])
            
            # Analyze OTM call activity (bullish speculation)
            call_analysis = self._analyze_option_group(otm_calls, 'ce', 'OTM_CALLS')
            
            # Analyze OTM put activity (bearish speculation)
            put_analysis = self._analyze_option_group(otm_puts, 'pe', 'OTM_PUTS')
            
            # Calculate speculative sentiment
            total_call_flow = call_analysis['total_weighted_flow']
            total_put_flow = put_analysis['total_weighted_flow']
            total_flow = total_call_flow + total_put_flow
            
            if total_flow > 0:
                speculative_sentiment = (total_call_flow - total_put_flow) / total_flow
            else:
                speculative_sentiment = 0.0
            
            # Speculative activity indicators
            speculative_indicators = {
                'lottery_ticket_buying': self._detect_lottery_ticket_activity(otm_calls, otm_puts),
                'volatility_chasing': self._detect_volatility_chasing(otm_calls, otm_puts),
                'expiry_concentration': self._analyze_expiry_concentration(otm_calls, otm_puts)
            }
            
            return {
                'call_analysis': call_analysis,
                'put_analysis': put_analysis,
                'speculative_sentiment': speculative_sentiment,
                'total_activity': total_flow,
                'speculative_indicators': speculative_indicators,
                'bullish_speculation': total_call_flow / total_flow if total_flow > 0 else 0.5,
                'bearish_speculation': total_put_flow / total_flow if total_flow > 0 else 0.5
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OTM sentiment: {e}")
            return {}
    
    def _analyze_option_group(self, 
                            options_df: pd.DataFrame,
                            option_prefix: str,
                            group_name: str) -> Dict[str, Any]:
        """Analyze a specific group of options"""
        try:
            if options_df.empty:
                return {
                    'total_oi': 0,
                    'total_volume': 0,
                    'total_weighted_flow': 0,
                    'avg_activity': 0,
                    'strike_count': 0,
                    'group_name': group_name
                }
            
            # Extract OI and Volume
            oi_column = f'{option_prefix}_oi'
            volume_column = f'{option_prefix}_volume'
            
            total_oi = options_df[oi_column].sum() if oi_column in options_df.columns else 0
            total_volume = options_df[volume_column].sum() if volume_column in options_df.columns else 0
            
            # Calculate weighted flow using dual weighting
            weighted_flow = self.oi_weight * total_oi + self.volume_weight * total_volume
            
            # Additional metrics
            strike_count = len(options_df)
            avg_activity = weighted_flow / strike_count if strike_count > 0 else 0
            
            # Activity concentration (Gini coefficient-like measure)
            if strike_count > 1:
                activities = [
                    self.oi_weight * row.get(oi_column, 0) + self.volume_weight * row.get(volume_column, 0)
                    for _, row in options_df.iterrows()
                ]
                concentration = self._calculate_concentration_index(activities)
            else:
                concentration = 1.0 if weighted_flow > 0 else 0.0
            
            return {
                'total_oi': total_oi,
                'total_volume': total_volume,
                'total_weighted_flow': weighted_flow,
                'avg_activity': avg_activity,
                'strike_count': strike_count,
                'activity_concentration': concentration,
                'group_name': group_name
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option group {group_name}: {e}")
            return {}
    
    def _analyze_option_flows(self,
                            market_data: pd.DataFrame,
                            moneyness_classification: Dict[str, pd.DataFrame],
                            spot_price: float) -> Dict[str, Any]:
        """Analyze overall option flows across moneyness categories"""
        try:
            flow_analysis = {}
            
            # Analyze flows for each category
            for category, options_df in moneyness_classification.items():
                if options_df.empty:
                    continue
                
                option_type = 'ce' if 'calls' in category else 'pe'
                flow_analysis[category] = self._analyze_option_group(options_df, option_type, category)
            
            # Calculate flow ratios
            total_call_flow = sum([
                analysis['total_weighted_flow'] 
                for cat, analysis in flow_analysis.items() 
                if 'calls' in cat
            ])
            
            total_put_flow = sum([
                analysis['total_weighted_flow'] 
                for cat, analysis in flow_analysis.items() 
                if 'puts' in cat
            ])
            
            total_flow = total_call_flow + total_put_flow
            
            # Flow distribution analysis
            flow_distribution = {
                'call_put_ratio': total_call_flow / total_put_flow if total_put_flow > 0 else float('inf'),
                'call_flow_percentage': total_call_flow / total_flow if total_flow > 0 else 0,
                'put_flow_percentage': total_put_flow / total_flow if total_flow > 0 else 0,
                'total_flow': total_flow
            }
            
            return {
                'category_analysis': flow_analysis,
                'flow_distribution': flow_distribution,
                'dominant_category': self._identify_dominant_category(flow_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing option flows: {e}")
            return {}
    
    def _analyze_time_patterns(self,
                             market_data: pd.DataFrame,
                             moneyness_classification: Dict[str, pd.DataFrame],
                             current_time: datetime) -> Dict[str, Any]:
        """Analyze time-based patterns in ITM/OTM activity"""
        try:
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # Time-of-day analysis
            time_analysis = {
                'market_session': self._classify_market_session(current_hour, current_minute),
                'activity_timing': self._analyze_activity_timing(current_time),
                'session_bias': self._calculate_session_bias(current_hour)
            }
            
            return time_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing time patterns: {e}")
            return {}
    
    def _calculate_sentiment_scores(self,
                                  itm_analysis: Dict[str, Any],
                                  otm_analysis: Dict[str, Any],
                                  flow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive sentiment scores"""
        try:
            # ITM sentiment (institutional)
            itm_sentiment = itm_analysis.get('net_sentiment', 0.0)
            
            # OTM sentiment (speculative)
            otm_sentiment = otm_analysis.get('speculative_sentiment', 0.0)
            
            # Combined sentiment with limited contribution
            itm_contribution = np.clip(itm_sentiment * self.max_itm_contribution, 
                                     -self.max_itm_contribution, self.max_itm_contribution)
            otm_contribution = np.clip(otm_sentiment * self.max_otm_contribution,
                                     -self.max_otm_contribution, self.max_otm_contribution)
            
            combined_sentiment = itm_contribution + otm_contribution
            
            return {
                'itm_sentiment': itm_sentiment,
                'otm_sentiment': otm_sentiment,
                'itm_contribution': itm_contribution,
                'otm_contribution': otm_contribution,
                'combined_sentiment': combined_sentiment,
                'sentiment_confidence': self._calculate_sentiment_confidence(itm_analysis, otm_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error calculating sentiment scores: {e}")
            return {}
    
    def _apply_risk_adjustments(self,
                              sentiment_scores: Dict[str, Any],
                              moneyness_classification: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Apply risk adjustments to sentiment scores"""
        try:
            # Risk adjustment factors
            concentration_risk = self._calculate_concentration_risk(moneyness_classification)
            liquidity_risk = self._calculate_liquidity_risk(moneyness_classification)
            
            # Adjust sentiment scores for risk
            risk_adjusted_sentiment = sentiment_scores.get('combined_sentiment', 0.0)
            
            # Reduce sentiment strength if high concentration or low liquidity
            risk_multiplier = (1.0 - concentration_risk * 0.3) * (1.0 - liquidity_risk * 0.2)
            risk_adjusted_sentiment *= risk_multiplier
            
            return {
                'risk_adjusted_sentiment': risk_adjusted_sentiment,
                'concentration_risk': concentration_risk,
                'liquidity_risk': liquidity_risk,
                'risk_multiplier': risk_multiplier,
                'original_sentiment': sentiment_scores.get('combined_sentiment', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error applying risk adjustments: {e}")
            return {}
    
    def _detect_institutional_activity(self,
                                     itm_analysis: Dict[str, Any],
                                     otm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect indicators of institutional trading activity"""
        try:
            indicators = {
                'high_itm_activity': itm_analysis.get('total_activity', 0) > self.institutional_size_threshold,
                'coordinated_strikes': self._detect_coordinated_activity(itm_analysis),
                'size_concentration': self._detect_size_concentration(itm_analysis, otm_analysis),
                'pattern_sophistication': self._detect_sophisticated_patterns(itm_analysis)
            }
            
            # Overall institutional activity score
            institutional_score = sum(indicators.values()) / len(indicators)
            
            return {
                'indicators': indicators,
                'institutional_score': institutional_score,
                'confidence': 'high' if institutional_score > 0.7 else 'medium' if institutional_score > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error detecting institutional activity: {e}")
            return {}
    
    # Helper methods for various calculations
    def _calculate_concentration_index(self, values: List[float]) -> float:
        """Calculate concentration index (simplified Gini coefficient)"""
        try:
            if not values or all(v == 0 for v in values):
                return 0.0
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            total = sum(sorted_values)
            
            if total == 0:
                return 0.0
            
            # Simplified Gini calculation
            concentration = sum((2 * i - n - 1) * v for i, v in enumerate(sorted_values, 1)) / (n * total)
            return abs(concentration)
            
        except:
            return 0.0
    
    def _classify_market_session(self, hour: int, minute: int) -> str:
        """Classify current market session"""
        if 9 <= hour < 10:
            return "market_open"
        elif 10 <= hour < 14:
            return "midday"
        elif 14 <= hour < 15:
            return "market_close"
        else:
            return "after_hours"
    
    def _get_moneyness_distribution(self, moneyness_classification: Dict[str, pd.DataFrame]) -> Dict[str, int]:
        """Get distribution of options across moneyness categories"""
        return {category: len(df) for category, df in moneyness_classification.items()}
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis result for error cases"""
        return {
            'itm_analysis': {},
            'otm_analysis': {},
            'flow_analysis': {},
            'time_analysis': {},
            'sentiment_scores': {
                'itm_sentiment': 0.0,
                'otm_sentiment': 0.0,
                'combined_sentiment': 0.0
            },
            'risk_adjusted_scores': {
                'risk_adjusted_sentiment': 0.0
            },
            'error': True
        }
    
    def _record_analysis(self, analysis_result: Dict[str, Any]):
        """Record analysis in history"""
        try:
            self.analysis_history.append({
                'timestamp': analysis_result['timestamp'],
                'spot_price': analysis_result['spot_price'],
                'itm_sentiment': analysis_result['sentiment_scores'].get('itm_sentiment', 0),
                'otm_sentiment': analysis_result['sentiment_scores'].get('otm_sentiment', 0),
                'combined_sentiment': analysis_result['sentiment_scores'].get('combined_sentiment', 0)
            })
            
            # Keep only last 100 analyses
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-100:]
                
        except Exception as e:
            logger.error(f"Error recording analysis: {e}")
    
    # Placeholder implementations for complex detection methods
    def _detect_large_blocks(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> bool:
        """Detect large block trading activity"""
        return False  # Placeholder
    
    def _detect_unusual_oi_buildup(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> bool:
        """Detect unusual OI buildup patterns"""
        return False  # Placeholder
    
    def _detect_cross_strike_patterns(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> bool:
        """Detect coordinated cross-strike patterns"""
        return False  # Placeholder
    
    def _detect_lottery_ticket_activity(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> bool:
        """Detect lottery ticket buying behavior"""
        return False  # Placeholder
    
    def _detect_volatility_chasing(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> bool:
        """Detect volatility chasing behavior"""
        return False  # Placeholder
    
    def _analyze_expiry_concentration(self, calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze concentration by expiry"""
        return {}  # Placeholder
    
    def _identify_dominant_category(self, flow_analysis: Dict[str, Dict]) -> str:
        """Identify the category with highest flow"""
        return "unknown"  # Placeholder
    
    def _analyze_activity_timing(self, current_time: datetime) -> Dict[str, Any]:
        """Analyze activity timing patterns"""
        return {}  # Placeholder
    
    def _calculate_session_bias(self, hour: int) -> float:
        """Calculate bias based on session time"""
        return 0.0  # Placeholder
    
    def _calculate_sentiment_confidence(self, itm_analysis: Dict, otm_analysis: Dict) -> float:
        """Calculate confidence in sentiment analysis"""
        return 0.5  # Placeholder
    
    def _calculate_concentration_risk(self, moneyness_classification: Dict) -> float:
        """Calculate concentration risk"""
        return 0.0  # Placeholder
    
    def _calculate_liquidity_risk(self, moneyness_classification: Dict) -> float:
        """Calculate liquidity risk"""
        return 0.0  # Placeholder
    
    def _detect_coordinated_activity(self, itm_analysis: Dict) -> bool:
        """Detect coordinated trading activity"""
        return False  # Placeholder
    
    def _detect_size_concentration(self, itm_analysis: Dict, otm_analysis: Dict) -> bool:
        """Detect size concentration patterns"""
        return False  # Placeholder
    
    def _detect_sophisticated_patterns(self, itm_analysis: Dict) -> bool:
        """Detect sophisticated trading patterns"""
        return False  # Placeholder