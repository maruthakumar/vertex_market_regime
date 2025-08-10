"""
Enhanced POS Processor with all advanced calculations
Handles all 200+ parameters and complex analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
import json

from .models_enhanced import (
    CompletePOSStrategy, EnhancedLegModel, AdjustmentRule,
    MarketRegime, BEAction, AdjustmentAction
)


@dataclass
class LegResult:
    """Result for a single leg"""
    leg_id: str
    leg_name: str
    strike: float
    price: float
    premium: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    oi: int
    volume: int
    breakeven_contribution: float = 0.0
    risk_contribution: float = 0.0
    
    
@dataclass
class PositionResult:
    """Result for complete position at a point in time"""
    trade_date: date
    trade_time: str
    spot: float
    # Aggregate metrics
    total_premium: float
    net_delta: float
    net_gamma: float
    net_theta: float
    net_vega: float
    # Breakeven
    upper_breakeven: float
    lower_breakeven: float
    be_probability: float
    distance_to_be: float
    # Risk metrics
    max_loss: float
    max_profit: float
    risk_reward_ratio: float
    probability_of_profit: float
    expected_value: float
    # Legs
    legs: List[LegResult]
    # Market conditions
    market_regime: Optional[MarketRegime] = None
    volatility_regime: Optional[str] = None
    trend_state: Optional[str] = None
    # Adjustments
    adjustment_triggered: bool = False
    adjustment_rules_triggered: List[str] = field(default_factory=list)
    adjustment_actions: List[Dict[str, Any]] = field(default_factory=list)
    

@dataclass
class AdjustmentEvent:
    """Record of an adjustment made"""
    date: date
    time: str
    rule_id: str
    rule_name: str
    trigger_type: str
    action_type: str
    leg_affected: Optional[str]
    old_strike: Optional[float]
    new_strike: Optional[float]
    cost: float
    be_improvement: float
    risk_reduction: float
    success: bool
    notes: str
    

@dataclass
class EnhancedPOSResults:
    """Complete results with all analytics"""
    # Core results
    strategy_summary: Dict[str, Any]
    position_results: List[PositionResult]
    adjustment_events: List[AdjustmentEvent]
    
    # Analytics
    daily_pnl: List[Dict[str, Any]]
    hourly_pnl: List[Dict[str, Any]]
    
    # Risk metrics
    risk_metrics: Dict[str, Any]
    greek_evolution: pd.DataFrame
    
    # Market structure
    market_analysis: Optional[pd.DataFrame]
    volatility_analysis: Optional[pd.DataFrame]
    
    # Breakeven analysis
    breakeven_analysis: pd.DataFrame
    be_scenarios: Dict[str, Any]
    
    # Performance metrics
    metrics: Dict[str, Any]
    
    # Adjustment analysis
    adjustment_summary: Dict[str, Any]
    adjustment_effectiveness: Dict[str, Any]
    
    # Optimization suggestions
    optimization_suggestions: List[Dict[str, Any]]


class EnhancedPOSProcessor:
    """Process POS strategy results with all advanced features"""
    
    def __init__(self):
        self.tick_size = 0.05
        self.lot_size = 50
        
    def process_results(self, df: pd.DataFrame, strategy: CompletePOSStrategy,
                       market_df: Optional[pd.DataFrame] = None,
                       volatility_df: Optional[pd.DataFrame] = None,
                       greek_df: Optional[pd.DataFrame] = None) -> EnhancedPOSResults:
        """Process all results with complete analytics"""
        
        # Process position data
        position_results = self._process_positions(df, strategy)
        
        # Track adjustments
        adjustment_events = self._simulate_adjustments(position_results, strategy)
        
        # Calculate P&L series
        daily_pnl = self._calculate_daily_pnl(position_results)
        hourly_pnl = self._calculate_hourly_pnl(position_results)
        
        # Risk analysis
        risk_metrics = self._calculate_risk_metrics(position_results, strategy)
        
        # Greek evolution
        greek_evolution = self._analyze_greek_evolution(position_results)
        
        # Market structure analysis
        market_analysis = None
        if market_df is not None and not market_df.empty:
            market_analysis = self._analyze_market_structure(market_df, strategy)
        
        # Volatility analysis
        volatility_analysis = None
        if volatility_df is not None and not volatility_df.empty:
            volatility_analysis = self._analyze_volatility(volatility_df, strategy)
        
        # Breakeven analysis
        breakeven_analysis, be_scenarios = self._analyze_breakeven(position_results, strategy)
        
        # Performance metrics
        metrics = self._calculate_performance_metrics(
            position_results, adjustment_events, strategy
        )
        
        # Adjustment analysis
        adjustment_summary = self._analyze_adjustments(adjustment_events)
        adjustment_effectiveness = self._calculate_adjustment_effectiveness(
            adjustment_events, position_results
        )
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            metrics, risk_metrics, adjustment_summary, strategy
        )
        
        # Create strategy summary
        strategy_summary = self._create_strategy_summary(strategy, metrics)
        
        return EnhancedPOSResults(
            strategy_summary=strategy_summary,
            position_results=position_results,
            adjustment_events=adjustment_events,
            daily_pnl=daily_pnl,
            hourly_pnl=hourly_pnl,
            risk_metrics=risk_metrics,
            greek_evolution=greek_evolution,
            market_analysis=market_analysis,
            volatility_analysis=volatility_analysis,
            breakeven_analysis=breakeven_analysis,
            be_scenarios=be_scenarios,
            metrics=metrics,
            adjustment_summary=adjustment_summary,
            adjustment_effectiveness=adjustment_effectiveness,
            optimization_suggestions=optimization_suggestions
        )
    
    def _process_positions(self, df: pd.DataFrame, strategy: CompletePOSStrategy) -> List[PositionResult]:
        """Process position data from query results"""
        
        results = []
        
        for _, row in df.iterrows():
            # Extract leg results
            legs = []
            for i, leg_model in enumerate(strategy.get_active_legs()):
                leg_result = LegResult(
                    leg_id=leg_model.leg_id,
                    leg_name=leg_model.leg_name,
                    strike=row.get(f'leg{i}_strike', 0),
                    price=row.get(f'leg{i}_price', 0),
                    premium=row.get(f'leg{i}_premium', 0),
                    delta=row.get(f'leg{i}_delta', 0),
                    gamma=row.get(f'leg{i}_gamma', 0),
                    theta=row.get(f'leg{i}_theta', 0),
                    vega=row.get(f'leg{i}_vega', 0),
                    iv=row.get(f'leg{i}_iv', 0),
                    oi=row.get(f'leg{i}_oi', 0),
                    volume=row.get(f'leg{i}_volume', 0)
                )
                
                # Calculate BE contribution
                if strategy.strategy.breakeven_config.enabled and leg_model.track_leg_be:
                    leg_result.breakeven_contribution = self._calculate_be_contribution(
                        leg_result, leg_model
                    )
                
                # Calculate risk contribution
                leg_result.risk_contribution = abs(leg_result.delta) / self.lot_size
                
                legs.append(leg_result)
            
            # Create position result
            position = PositionResult(
                trade_date=row['trade_date'],
                trade_time=row['trade_time'],
                spot=row['spot'],
                total_premium=row.get('total_premium', 0),
                net_delta=row.get('net_delta', 0),
                net_gamma=row.get('net_gamma', 0),
                net_theta=row.get('net_theta', 0),
                net_vega=row.get('net_vega', 0),
                upper_breakeven=row.get('upper_be', row['spot']),
                lower_breakeven=row.get('lower_be', row['spot']),
                be_probability=row.get('be_probability', 0.5),
                distance_to_be=self._calculate_distance_to_be(row),
                max_loss=self._calculate_max_loss(legs, strategy),
                max_profit=self._calculate_max_profit(legs, strategy),
                risk_reward_ratio=0,  # Will calculate
                probability_of_profit=row.get('be_probability', 0.5),
                expected_value=0,  # Will calculate
                legs=legs,
                market_regime=self._determine_market_regime(row),
                volatility_regime=row.get('vol_regime'),
                trend_state=row.get('trend_state')
            )
            
            # Calculate derived metrics
            if position.max_loss != 0:
                position.risk_reward_ratio = abs(position.max_profit / position.max_loss)
            
            position.expected_value = self._calculate_expected_value(position)
            
            # Check for adjustments
            if strategy.adjustment_rules:
                triggered_rules = self._check_adjustment_triggers(position, strategy)
                position.adjustment_triggered = len(triggered_rules) > 0
                position.adjustment_rules_triggered = [r.rule_id for r in triggered_rules]
                position.adjustment_actions = self._determine_adjustment_actions(
                    triggered_rules, position
                )
            
            results.append(position)
        
        return results
    
    def _simulate_adjustments(self, positions: List[PositionResult], 
                            strategy: CompletePOSStrategy) -> List[AdjustmentEvent]:
        """Simulate adjustments based on rules"""
        
        events = []
        
        if not strategy.adjustment_rules:
            return events
        
        # Track adjustment state
        last_adjustment_time = {}
        adjustment_count = {}
        
        for position in positions:
            if not position.adjustment_triggered:
                continue
            
            for rule_id in position.adjustment_rules_triggered:
                rule = next((r for r in strategy.adjustment_rules if r.rule_id == rule_id), None)
                if not rule:
                    continue
                
                # Check cooldown
                if rule_id in last_adjustment_time:
                    time_since_last = (position.trade_date - last_adjustment_time[rule_id]).days
                    if time_since_last * 1440 < rule.adjustment_cooldown:
                        continue
                
                # Check max adjustments
                if rule_id not in adjustment_count:
                    adjustment_count[rule_id] = 0
                if rule.max_adjustments and adjustment_count[rule_id] >= rule.max_adjustments:
                    continue
                
                # Simulate adjustment
                event = self._simulate_single_adjustment(position, rule, strategy)
                if event.success:
                    events.append(event)
                    last_adjustment_time[rule_id] = position.trade_date
                    adjustment_count[rule_id] += 1
        
        return events
    
    def _calculate_daily_pnl(self, positions: List[PositionResult]) -> List[Dict[str, Any]]:
        """Calculate daily P&L"""
        
        daily_pnl = []
        
        # Group by date
        df = pd.DataFrame([{
            'date': p.trade_date,
            'premium': p.total_premium,
            'delta': p.net_delta,
            'theta': p.net_theta
        } for p in positions])
        
        if df.empty:
            return daily_pnl
        
        grouped = df.groupby('date').agg({
            'premium': 'last',
            'delta': 'last',
            'theta': 'mean'
        })
        
        # Calculate daily changes
        grouped['daily_pnl'] = grouped['premium'].diff()
        grouped['cumulative_pnl'] = grouped['premium'].cumsum()
        
        for date, row in grouped.iterrows():
            daily_pnl.append({
                'date': date,
                'daily_pnl': row['daily_pnl'] if pd.notna(row['daily_pnl']) else 0,
                'cumulative_pnl': row['cumulative_pnl'],
                'position_delta': row['delta'],
                'theta_collected': row['theta']
            })
        
        return daily_pnl
    
    def _calculate_hourly_pnl(self, positions: List[PositionResult]) -> List[Dict[str, Any]]:
        """Calculate hourly P&L for intraday analysis"""
        
        hourly_pnl = []
        
        # Group by date and hour
        for position in positions:
            hour = position.trade_time.split(':')[0]
            hourly_pnl.append({
                'date': position.trade_date,
                'hour': hour,
                'premium': position.total_premium,
                'delta': position.net_delta,
                'gamma': position.net_gamma,
                'theta': position.net_theta,
                'vega': position.net_vega
            })
        
        return hourly_pnl
    
    def _calculate_risk_metrics(self, positions: List[PositionResult], 
                              strategy: CompletePOSStrategy) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        if not positions:
            return {}
        
        # Extract metrics
        premiums = [p.total_premium for p in positions]
        deltas = [p.net_delta for p in positions]
        gammas = [p.net_gamma for p in positions]
        thetas = [p.net_theta for p in positions]
        vegas = [p.net_vega for p in positions]
        
        # Calculate VaR if enabled
        var_95 = None
        var_99 = None
        if strategy.strategy.risk_management.use_var:
            confidence = strategy.strategy.risk_management.var_confidence
            lookback = strategy.strategy.risk_management.var_lookback
            
            if len(premiums) > lookback:
                returns = pd.Series(premiums).pct_change().dropna()
                var_95 = np.percentile(returns, 5) * strategy.portfolio.position_size_value
                var_99 = np.percentile(returns, 1) * strategy.portfolio.position_size_value
        
        # Calculate Kelly fraction if enabled
        kelly_fraction = None
        if strategy.strategy.risk_management.use_kelly_criterion:
            wins = [p for p in premiums if p > 0]
            losses = [p for p in premiums if p <= 0]
            if wins and losses:
                win_rate = len(wins) / len(premiums)
                avg_win = np.mean(wins)
                avg_loss = abs(np.mean(losses))
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = min(kelly_fraction, strategy.strategy.risk_management.kelly_fraction)
        
        return {
            # Basic risk metrics
            'max_drawdown': self._calculate_max_drawdown(premiums),
            'max_drawdown_duration': self._calculate_max_drawdown_duration(positions),
            'volatility': np.std(premiums) if len(premiums) > 1 else 0,
            'downside_deviation': self._calculate_downside_deviation(premiums),
            'sortino_ratio': self._calculate_sortino_ratio(premiums),
            'calmar_ratio': self._calculate_calmar_ratio(premiums),
            
            # Greek risk metrics
            'avg_delta_exposure': np.mean(np.abs(deltas)),
            'max_delta_exposure': np.max(np.abs(deltas)),
            'avg_gamma_exposure': np.mean(np.abs(gammas)),
            'max_gamma_exposure': np.max(np.abs(gammas)),
            'avg_theta_collection': np.mean(thetas),
            'avg_vega_exposure': np.mean(np.abs(vegas)),
            
            # Advanced risk metrics
            'var_95': var_95,
            'var_99': var_99,
            'kelly_fraction': kelly_fraction,
            'risk_adjusted_return': self._calculate_risk_adjusted_return(premiums),
            
            # Concentration risk
            'position_concentration': self._calculate_position_concentration(positions),
            'strike_concentration': self._calculate_strike_concentration(positions),
            
            # Tail risk
            'left_tail_risk': self._calculate_tail_risk(premiums, 'left'),
            'right_tail_risk': self._calculate_tail_risk(premiums, 'right'),
            
            # Correlation risk
            'leg_correlation': self._calculate_leg_correlation(positions),
            
            # BE risk
            'avg_be_distance': np.mean([p.distance_to_be for p in positions]),
            'min_be_distance': np.min([p.distance_to_be for p in positions]),
            'be_breach_count': sum(1 for p in positions if p.distance_to_be < 0)
        }
    
    def _analyze_greek_evolution(self, positions: List[PositionResult]) -> pd.DataFrame:
        """Analyze how Greeks evolve over time"""
        
        data = []
        for position in positions:
            data.append({
                'datetime': f"{position.trade_date} {position.trade_time}",
                'delta': position.net_delta,
                'gamma': position.net_gamma,
                'theta': position.net_theta,
                'vega': position.net_vega,
                'speed': self._calculate_speed(position),  # Rate of change of gamma
                'charm': self._calculate_charm(position),  # Rate of change of delta over time
                'vanna': self._calculate_vanna(position),  # Rate of change of delta with vol
                'color': self._calculate_color(position)   # Rate of change of gamma over time
            })
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        # Add rolling statistics
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            df[f'{greek}_ma'] = df[greek].rolling(window=20).mean()
            df[f'{greek}_std'] = df[greek].rolling(window=20).std()
        
        return df
    
    def _analyze_market_structure(self, market_df: pd.DataFrame, 
                                strategy: CompletePOSStrategy) -> pd.DataFrame:
        """Analyze market structure data"""
        
        if not strategy.market_structure or not strategy.market_structure.enabled:
            return market_df
        
        # Add regime classification
        market_df['regime'] = market_df.apply(
            lambda row: self._classify_market_regime(row), axis=1
        )
        
        # Add support/resistance levels
        if strategy.market_structure.detect_sr_levels:
            market_df = self._add_sr_levels(market_df, strategy.market_structure)
        
        # Add volume profile
        if strategy.market_structure.analyze_volume:
            market_df = self._add_volume_profile(market_df)
        
        # Add microstructure metrics
        if strategy.market_structure.analyze_bid_ask_spread:
            market_df = self._add_microstructure_metrics(market_df)
        
        return market_df
    
    def _analyze_volatility(self, volatility_df: pd.DataFrame, 
                          strategy: CompletePOSStrategy) -> pd.DataFrame:
        """Analyze volatility metrics"""
        
        vol_filter = strategy.strategy.volatility_filter
        
        # Add volatility regime classification
        volatility_df['vol_regime'] = pd.cut(
            volatility_df['iv_percentile'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['VERY_LOW', 'LOW', 'MEDIUM', 'HIGH', 'VERY_HIGH']
        )
        
        # Add term structure analysis
        if 'cw_iv' in volatility_df.columns and 'cm_iv' in volatility_df.columns:
            volatility_df['term_structure'] = volatility_df['cm_iv'] - volatility_df['cw_iv']
            volatility_df['term_structure_state'] = volatility_df['term_structure'].apply(
                lambda x: 'CONTANGO' if x > 0 else 'BACKWARDATION'
            )
        
        # Add IV-HV analysis
        if 'hv_30' in volatility_df.columns:
            volatility_df['iv_premium'] = volatility_df['atm_iv'] - volatility_df['hv_30']
            volatility_df['iv_premium_percentile'] = volatility_df['iv_premium'].rank(pct=True)
        
        # Add volatility smile analysis
        if 'otm_put_iv' in volatility_df.columns and 'otm_call_iv' in volatility_df.columns:
            volatility_df['put_call_skew'] = volatility_df['otm_put_iv'] - volatility_df['otm_call_iv']
            volatility_df['smile_slope'] = (volatility_df['otm_call_iv'] - volatility_df['otm_put_iv']) / 2
        
        return volatility_df
    
    def _analyze_breakeven(self, positions: List[PositionResult], 
                         strategy: CompletePOSStrategy) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Analyze breakeven dynamics"""
        
        be_data = []
        for position in positions:
            be_data.append({
                'datetime': f"{position.trade_date} {position.trade_time}",
                'spot': position.spot,
                'upper_be': position.upper_breakeven,
                'lower_be': position.lower_breakeven,
                'distance_to_upper': position.spot - position.upper_breakeven,
                'distance_to_lower': position.lower_breakeven - position.spot,
                'min_distance': position.distance_to_be,
                'be_probability': position.be_probability,
                'time_to_expiry': self._calculate_time_to_expiry(position, strategy),
                'be_status': self._classify_be_status(position, strategy)
            })
        
        be_df = pd.DataFrame(be_data)
        be_df['datetime'] = pd.to_datetime(be_df['datetime'])
        be_df.set_index('datetime', inplace=True)
        
        # Breakeven scenarios
        be_scenarios = {
            'base_case': self._calculate_be_scenario(positions, 'base'),
            'stress_up': self._calculate_be_scenario(positions, 'stress_up', 0.1),
            'stress_down': self._calculate_be_scenario(positions, 'stress_down', -0.1),
            'high_vol': self._calculate_be_scenario(positions, 'high_vol'),
            'low_vol': self._calculate_be_scenario(positions, 'low_vol'),
            'time_decay': self._calculate_be_scenario(positions, 'time_decay')
        }
        
        return be_df, be_scenarios
    
    def _calculate_performance_metrics(self, positions: List[PositionResult],
                                     adjustments: List[AdjustmentEvent],
                                     strategy: CompletePOSStrategy) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not positions:
            return {}
        
        premiums = [p.total_premium for p in positions]
        
        # Basic metrics
        total_trades = len(positions)
        winning_trades = sum(1 for p in premiums if p > 0)
        losing_trades = sum(1 for p in premiums if p <= 0)
        
        # P&L metrics
        gross_pnl = sum(premiums)
        adjustment_costs = sum(a.cost for a in adjustments)
        commission_costs = total_trades * strategy.portfolio.transaction_costs * strategy.portfolio.position_size_value
        net_pnl = gross_pnl - adjustment_costs - commission_costs
        
        # Calculate advanced metrics
        metrics = {
            # Basic metrics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            
            # P&L metrics
            'gross_pnl': gross_pnl,
            'adjustment_costs': adjustment_costs,
            'commission_costs': commission_costs,
            'net_pnl': net_pnl,
            'avg_trade_pnl': net_pnl / total_trades if total_trades > 0 else 0,
            
            # Return metrics
            'total_return': net_pnl / strategy.portfolio.initial_capital,
            'annualized_return': self._calculate_annualized_return(net_pnl, positions, strategy),
            'sharpe_ratio': self._calculate_sharpe_ratio(premiums),
            'sortino_ratio': self._calculate_sortino_ratio(premiums),
            
            # Risk metrics
            'max_consecutive_wins': self._calculate_max_consecutive(premiums, True),
            'max_consecutive_losses': self._calculate_max_consecutive(premiums, False),
            'profit_factor': self._calculate_profit_factor(premiums),
            'expectancy': self._calculate_expectancy(premiums),
            
            # Greek metrics
            'avg_theta_collected': np.mean([p.net_theta for p in positions]),
            'total_theta_collected': sum(p.net_theta for p in positions),
            'theta_efficiency': self._calculate_theta_efficiency(positions),
            
            # Adjustment metrics
            'total_adjustments': len(adjustments),
            'adjustment_success_rate': sum(1 for a in adjustments if a.success) / len(adjustments) if adjustments else 0,
            'avg_adjustment_cost': np.mean([a.cost for a in adjustments]) if adjustments else 0,
            'adjustment_roi': self._calculate_adjustment_roi(adjustments, positions),
            
            # BE metrics
            'be_breach_rate': sum(1 for p in positions if p.distance_to_be < 0) / len(positions),
            'avg_be_distance': np.mean([p.distance_to_be for p in positions]),
            'be_efficiency': self._calculate_be_efficiency(positions),
            
            # Time-based metrics
            'avg_time_in_trade': self._calculate_avg_time_in_trade(positions),
            'time_in_profit': self._calculate_time_in_profit(positions),
            'overnight_risk': self._calculate_overnight_risk(positions),
            
            # Capital efficiency
            'capital_turnover': self._calculate_capital_turnover(positions, strategy),
            'return_on_margin': self._calculate_return_on_margin(net_pnl, positions, strategy)
        }
        
        return metrics
    
    def _analyze_adjustments(self, events: List[AdjustmentEvent]) -> Dict[str, Any]:
        """Analyze adjustment patterns and effectiveness"""
        
        if not events:
            return {
                'total_adjustments': 0,
                'by_trigger_type': {},
                'by_action_type': {},
                'success_rate': 0,
                'avg_cost': 0,
                'avg_be_improvement': 0
            }
        
        # Group by trigger type
        by_trigger = {}
        for event in events:
            if event.trigger_type not in by_trigger:
                by_trigger[event.trigger_type] = []
            by_trigger[event.trigger_type].append(event)
        
        # Group by action type
        by_action = {}
        for event in events:
            if event.action_type not in by_action:
                by_action[event.action_type] = []
            by_action[event.action_type].append(event)
        
        # Calculate statistics
        trigger_stats = {}
        for trigger, trigger_events in by_trigger.items():
            trigger_stats[trigger] = {
                'count': len(trigger_events),
                'success_rate': sum(1 for e in trigger_events if e.success) / len(trigger_events),
                'avg_cost': np.mean([e.cost for e in trigger_events]),
                'avg_be_improvement': np.mean([e.be_improvement for e in trigger_events])
            }
        
        action_stats = {}
        for action, action_events in by_action.items():
            action_stats[action] = {
                'count': len(action_events),
                'success_rate': sum(1 for e in action_events if e.success) / len(action_events),
                'avg_cost': np.mean([e.cost for e in action_events]),
                'avg_risk_reduction': np.mean([e.risk_reduction for e in action_events])
            }
        
        return {
            'total_adjustments': len(events),
            'by_trigger_type': trigger_stats,
            'by_action_type': action_stats,
            'success_rate': sum(1 for e in events if e.success) / len(events),
            'avg_cost': np.mean([e.cost for e in events]),
            'avg_be_improvement': np.mean([e.be_improvement for e in events]),
            'total_cost': sum(e.cost for e in events),
            'most_common_trigger': max(by_trigger.items(), key=lambda x: len(x[1]))[0] if by_trigger else None,
            'most_common_action': max(by_action.items(), key=lambda x: len(x[1]))[0] if by_action else None
        }
    
    def _calculate_adjustment_effectiveness(self, events: List[AdjustmentEvent],
                                          positions: List[PositionResult]) -> Dict[str, Any]:
        """Calculate how effective adjustments were"""
        
        if not events:
            return {}
        
        # Track P&L before and after adjustments
        adjustment_impact = []
        
        for event in events:
            # Find positions around adjustment date
            before_positions = [p for p in positions if p.trade_date < event.date]
            after_positions = [p for p in positions if p.trade_date >= event.date]
            
            if before_positions and after_positions:
                avg_pnl_before = np.mean([p.total_premium for p in before_positions[-5:]])
                avg_pnl_after = np.mean([p.total_premium for p in after_positions[:5]])
                
                adjustment_impact.append({
                    'rule_id': event.rule_id,
                    'pnl_improvement': avg_pnl_after - avg_pnl_before - event.cost,
                    'be_improvement': event.be_improvement,
                    'risk_reduction': event.risk_reduction,
                    'cost': event.cost,
                    'net_benefit': (avg_pnl_after - avg_pnl_before) - event.cost
                })
        
        if not adjustment_impact:
            return {}
        
        # Calculate aggregate statistics
        df = pd.DataFrame(adjustment_impact)
        
        return {
            'avg_pnl_improvement': df['pnl_improvement'].mean(),
            'avg_net_benefit': df['net_benefit'].mean(),
            'positive_impact_rate': (df['net_benefit'] > 0).mean(),
            'total_net_benefit': df['net_benefit'].sum(),
            'cost_benefit_ratio': df['net_benefit'].sum() / df['cost'].sum() if df['cost'].sum() > 0 else 0,
            'best_performing_rule': df.groupby('rule_id')['net_benefit'].sum().idxmax() if not df.empty else None,
            'worst_performing_rule': df.groupby('rule_id')['net_benefit'].sum().idxmin() if not df.empty else None
        }
    
    def _generate_optimization_suggestions(self, metrics: Dict[str, Any],
                                         risk_metrics: Dict[str, Any],
                                         adjustment_summary: Dict[str, Any],
                                         strategy: CompletePOSStrategy) -> List[Dict[str, Any]]:
        """Generate optimization suggestions based on performance"""
        
        suggestions = []
        
        # Win rate optimization
        if metrics.get('win_rate', 0) < 0.6:
            suggestions.append({
                'category': 'WIN_RATE',
                'priority': 'HIGH',
                'suggestion': 'Consider tightening entry criteria or using volatility filters',
                'details': f"Current win rate: {metrics.get('win_rate', 0):.2%}",
                'potential_impact': 'Could improve win rate by 10-15%'
            })
        
        # Risk management
        if risk_metrics.get('max_drawdown', 0) > 0.2:
            suggestions.append({
                'category': 'RISK',
                'priority': 'HIGH',
                'suggestion': 'Implement stricter position sizing or stop-loss rules',
                'details': f"Max drawdown: {risk_metrics.get('max_drawdown', 0):.2%}",
                'potential_impact': 'Could reduce drawdown by 30-40%'
            })
        
        # Greek management
        if risk_metrics.get('avg_delta_exposure', 0) > 100:
            suggestions.append({
                'category': 'GREEKS',
                'priority': 'MEDIUM',
                'suggestion': 'Consider delta-neutral adjustments or hedging',
                'details': f"Average delta exposure: {risk_metrics.get('avg_delta_exposure', 0):.0f}",
                'potential_impact': 'Could reduce directional risk'
            })
        
        # Adjustment efficiency
        if adjustment_summary.get('success_rate', 0) < 0.7:
            suggestions.append({
                'category': 'ADJUSTMENTS',
                'priority': 'MEDIUM',
                'suggestion': 'Review adjustment triggers and thresholds',
                'details': f"Adjustment success rate: {adjustment_summary.get('success_rate', 0):.2%}",
                'potential_impact': 'Could improve adjustment ROI'
            })
        
        # Capital efficiency
        if metrics.get('capital_turnover', 0) < 2:
            suggestions.append({
                'category': 'CAPITAL',
                'priority': 'LOW',
                'suggestion': 'Consider increasing position frequency or size',
                'details': f"Capital turnover: {metrics.get('capital_turnover', 0):.2f}x",
                'potential_impact': 'Could improve capital utilization'
            })
        
        # Volatility regime
        if strategy.strategy.volatility_filter.use_ivp:
            suggestions.append({
                'category': 'VOLATILITY',
                'priority': 'MEDIUM',
                'suggestion': 'Fine-tune IVP entry thresholds based on backtest',
                'details': f"Current IVP range: {strategy.strategy.volatility_filter.ivp_min_entry:.0%}-{strategy.strategy.volatility_filter.ivp_max_entry:.0%}",
                'potential_impact': 'Could improve entry timing'
            })
        
        return suggestions
    
    # Helper calculation methods
    
    def _calculate_be_contribution(self, leg_result: LegResult, 
                                 leg_model: EnhancedLegModel) -> float:
        """Calculate leg's contribution to breakeven"""
        contribution = leg_result.premium / self.lot_size
        
        if leg_model.leg_be_contribution == 'NEGATIVE':
            contribution *= -1
        elif leg_model.leg_be_contribution == 'NEUTRAL':
            contribution = 0
        
        return contribution * leg_model.leg_be_weight
    
    def _calculate_distance_to_be(self, row: Dict[str, Any]) -> float:
        """Calculate minimum distance to breakeven"""
        upper_distance = abs(row['spot'] - row.get('upper_be', row['spot']))
        lower_distance = abs(row['spot'] - row.get('lower_be', row['spot']))
        return min(upper_distance, lower_distance)
    
    def _calculate_max_loss(self, legs: List[LegResult], 
                          strategy: CompletePOSStrategy) -> float:
        """Calculate maximum possible loss"""
        # For credit spreads, max loss is width - credit
        # For debit spreads, max loss is debit paid
        total_debit = sum(leg.premium for leg in legs if leg.premium < 0)
        total_credit = sum(leg.premium for leg in legs if leg.premium > 0)
        
        if total_credit > abs(total_debit):
            # Net credit position
            # Calculate spread width
            strikes = [leg.strike for leg in legs]
            if len(strikes) >= 2:
                width = (max(strikes) - min(strikes)) * self.lot_size
                return width - total_credit
            else:
                return abs(total_debit)
        else:
            # Net debit position
            return abs(total_debit)
    
    def _calculate_max_profit(self, legs: List[LegResult], 
                            strategy: CompletePOSStrategy) -> float:
        """Calculate maximum possible profit"""
        total_credit = sum(leg.premium for leg in legs if leg.premium > 0)
        total_debit = sum(leg.premium for leg in legs if leg.premium < 0)
        
        if total_credit > abs(total_debit):
            # Net credit position - max profit is credit received
            return total_credit + total_debit
        else:
            # Net debit position - calculate based on strategy type
            if strategy.strategy.strategy_subtype == 'CALENDAR_SPREAD':
                # Calendar spread max profit is theoretically unlimited
                return abs(total_debit) * 3  # Estimate 3x
            else:
                # Other debit spreads
                strikes = [leg.strike for leg in legs]
                if len(strikes) >= 2:
                    width = (max(strikes) - min(strikes)) * self.lot_size
                    return width - abs(total_debit)
                else:
                    return abs(total_debit) * 2  # Estimate 2x
    
    def _calculate_expected_value(self, position: PositionResult) -> float:
        """Calculate expected value of position"""
        # Simplified calculation
        ev = (position.max_profit * position.probability_of_profit + 
              position.max_loss * (1 - position.probability_of_profit))
        return ev
    
    def _determine_market_regime(self, row: Dict[str, Any]) -> Optional[MarketRegime]:
        """Determine market regime from data"""
        trend = row.get('trend_state', 'NEUTRAL')
        vol = row.get('volatility_regime', 'NORMAL_VOL')
        
        if 'STRONG_UP' in trend:
            return MarketRegime.TRENDING_UP
        elif 'STRONG_DOWN' in trend:
            return MarketRegime.TRENDING_DOWN
        elif vol == 'HIGH_VOL':
            return MarketRegime.HIGH_VOLATILITY
        elif vol == 'LOW_VOL':
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGE_BOUND
    
    def _check_adjustment_triggers(self, position: PositionResult,
                                  strategy: CompletePOSStrategy) -> List[AdjustmentRule]:
        """Check which adjustment rules are triggered"""
        triggered = []
        
        for rule in strategy.adjustment_rules:
            if not rule.enabled:
                continue
            
            # Implement trigger checking logic based on rule type
            # This is simplified - real implementation would be more complex
            
            if rule.trigger_type == 'GREEK_BASED':
                if rule.delta_trigger and abs(position.net_delta) > rule.delta_trigger:
                    triggered.append(rule)
                elif rule.gamma_trigger and abs(position.net_gamma) > rule.gamma_trigger:
                    triggered.append(rule)
                    
            elif rule.trigger_type == 'BE_BASED':
                if position.distance_to_be < rule.trigger_value:
                    triggered.append(rule)
        
        return triggered
    
    def _determine_adjustment_actions(self, rules: List[AdjustmentRule],
                                    position: PositionResult) -> List[Dict[str, Any]]:
        """Determine what adjustment actions to take"""
        actions = []
        
        for rule in rules:
            action = {
                'rule_id': rule.rule_id,
                'action_type': rule.action_type,
                'leg_id': rule.action_leg_id,
                'priority': rule.priority,
                'estimated_cost': self._estimate_adjustment_cost(rule, position),
                'estimated_improvement': self._estimate_adjustment_improvement(rule, position)
            }
            actions.append(action)
        
        # Sort by priority
        actions.sort(key=lambda x: x['priority'])
        
        return actions
    
    def _simulate_single_adjustment(self, position: PositionResult,
                                  rule: AdjustmentRule,
                                  strategy: CompletePOSStrategy) -> AdjustmentEvent:
        """Simulate a single adjustment"""
        
        # This is a simplified simulation
        cost = self._estimate_adjustment_cost(rule, position)
        be_improvement = self._estimate_adjustment_improvement(rule, position)
        risk_reduction = 0.1  # Simplified
        
        # Determine success based on market conditions
        success = True  # Simplified - always succeed for now
        
        return AdjustmentEvent(
            date=position.trade_date,
            time=position.trade_time,
            rule_id=rule.rule_id,
            rule_name=rule.rule_name,
            trigger_type=rule.trigger_type.value,
            action_type=rule.action_type.value,
            leg_affected=rule.action_leg_id,
            old_strike=position.legs[0].strike if position.legs else None,
            new_strike=None,  # Would be calculated
            cost=cost,
            be_improvement=be_improvement,
            risk_reduction=risk_reduction,
            success=success,
            notes=f"Triggered by {rule.trigger_type.value}"
        )
    
    def _estimate_adjustment_cost(self, rule: AdjustmentRule,
                                position: PositionResult) -> float:
        """Estimate cost of adjustment"""
        # Simplified estimation
        base_cost = 50  # Base transaction cost
        
        if rule.action_type in ['ROLL_STRIKE', 'ROLL_EXPIRY']:
            # Rolling involves closing and opening
            base_cost *= 2
        
        # Add slippage
        slippage = position.total_premium * 0.01
        
        return base_cost + slippage
    
    def _estimate_adjustment_improvement(self, rule: AdjustmentRule,
                                       position: PositionResult) -> float:
        """Estimate breakeven improvement from adjustment"""
        # Simplified estimation
        if rule.action_type == 'ROLL_STRIKE':
            return 50  # Points
        elif rule.action_type == 'ADD_HEDGE':
            return 100  # Points
        else:
            return 25  # Points
    
    def _calculate_max_drawdown(self, premiums: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not premiums:
            return 0
        
        cumulative = np.cumsum(premiums)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_max_drawdown_duration(self, positions: List[PositionResult]) -> int:
        """Calculate maximum drawdown duration in days"""
        if not positions:
            return 0
        
        premiums = [p.total_premium for p in positions]
        dates = [p.trade_date for p in positions]
        
        cumulative = np.cumsum(premiums)
        running_max = np.maximum.accumulate(cumulative)
        
        # Find drawdown periods
        in_drawdown = cumulative < running_max
        
        max_duration = 0
        current_duration = 0
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_duration += 1
            else:
                max_duration = max(max_duration, current_duration)
                current_duration = 0
        
        return max_duration
    
    def _calculate_downside_deviation(self, returns: List[float]) -> float:
        """Calculate downside deviation"""
        if not returns:
            return 0
        
        negative_returns = [r for r in returns if r < 0]
        if not negative_returns:
            return 0
        
        return np.std(negative_returns)
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free: float = 0) -> float:
        """Calculate Sortino ratio"""
        if not returns:
            return 0
        
        excess_returns = [r - risk_free for r in returns]
        avg_excess = np.mean(excess_returns)
        downside_dev = self._calculate_downside_deviation(returns)
        
        if downside_dev == 0:
            return 0
        
        return avg_excess / downside_dev * np.sqrt(252)  # Annualized
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free: float = 0) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        excess_returns = [r - risk_free for r in returns]
        avg_excess = np.mean(excess_returns)
        std_returns = np.std(returns)
        
        if std_returns == 0:
            return 0
        
        return avg_excess / std_returns * np.sqrt(252)  # Annualized
    
    def _calculate_calmar_ratio(self, returns: List[float]) -> float:
        """Calculate Calmar ratio"""
        if not returns:
            return 0
        
        total_return = sum(returns)
        max_dd = self._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0
        
        # Annualize
        days = len(returns)
        annual_return = total_return * 252 / days
        
        return annual_return / max_dd
    
    def _create_strategy_summary(self, strategy: CompletePOSStrategy,
                               metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive strategy summary"""
        
        return {
            'strategy_name': strategy.strategy.strategy_name,
            'strategy_type': f"{strategy.strategy.position_type.value} {strategy.strategy.strategy_subtype.value}",
            'start_date': str(strategy.portfolio.start_date),
            'end_date': str(strategy.portfolio.end_date),
            'legs': len(strategy.get_active_legs()),
            'adjustment_rules': len(strategy.adjustment_rules) if strategy.adjustment_rules else 0,
            'capital': strategy.portfolio.initial_capital,
            'position_size': strategy.portfolio.position_size_value,
            'net_pnl': metrics.get('net_pnl', 0),
            'total_return': metrics.get('total_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'win_rate': metrics.get('win_rate', 0)
        }
    
    # Additional helper methods for advanced calculations...
    
    def _calculate_speed(self, position: PositionResult) -> float:
        """Calculate speed (3rd order Greek)"""
        # Simplified - would need more data for accurate calculation
        return 0
    
    def _calculate_charm(self, position: PositionResult) -> float:
        """Calculate charm (delta decay)"""
        # Simplified
        return -position.net_delta * 0.01
    
    def _calculate_vanna(self, position: PositionResult) -> float:
        """Calculate vanna (delta/vega cross)"""
        # Simplified
        return position.net_delta * position.net_vega * 0.0001
    
    def _calculate_color(self, position: PositionResult) -> float:
        """Calculate color (gamma decay)"""
        # Simplified
        return -position.net_gamma * 0.01