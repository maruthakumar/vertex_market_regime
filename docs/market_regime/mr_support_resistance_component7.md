# Component 7: Support & Resistance Formation Logic
## Advanced Dynamic Levels Detection with Dual DTE Learning

### Overview

Component 7 represents the **dynamic support and resistance detection engine** that analyzes both rolling straddle prices and underlying prices to identify critical levels that adapt to market structure changes. This system employs comprehensive historical learning with dual DTE analysis and dynamic weight adjustment based on performance feedback.

**Revolutionary Approach**: Unlike static support/resistance levels, this system uses **adaptive level formation** that learns from historical price reactions and adjusts level significance based on DTE-specific and market-condition-specific performance patterns.

---

## Core Architecture

### Multi-Asset Support/Resistance Framework

```python
class DynamicSupportResistanceEngine:
    def __init__(self):
        # Dual Asset Analysis Framework
        self.asset_types = {
            'rolling_straddle_prices': {
                'component_1_atm_straddle': {'weight': 0.35, 'strikes': ['ATM']},
                'component_1_itm1_straddle': {'weight': 0.20, 'strikes': ['ITM1']}, 
                'component_1_otm1_straddle': {'weight': 0.20, 'strikes': ['OTM1']},
                'component_3_cumulative_straddles': {'weight': 0.25, 'strikes': ['ATM±7']}
            },
            'underlying_prices': {
                'daily_levels': {'weight': 0.40, 'timeframe': 'daily'},
                'weekly_levels': {'weight': 0.35, 'timeframe': 'weekly'},
                'monthly_levels': {'weight': 0.25, 'timeframe': 'monthly'}
            }
        }
        
        # Dual DTE Analysis Framework
        self.specific_dte_levels = {
            f'dte_{i}': {
                'historical_levels': deque(maxlen=252),  # 1 year per DTE
                'level_strength_patterns': {},
                'breakout_success_rates': {},
                'false_breakout_rates': {},
                'learned_level_weights': {},
                'performance_metrics': {}
            } for i in range(91)  # DTE 0 to 90
        }
        
        # DTE Range Analysis
        self.dte_range_levels = {
            'dte_0_to_7': {
                'range': (0, 7),
                'label': 'Weekly expiry cycle levels',
                'historical_levels': deque(maxlen=1260),  # 5 years
                'level_formation_patterns': {},
                'dynamic_weights': {},
                'market_structure_adaptations': {}
            },
            'dte_8_to_30': {
                'range': (8, 30),
                'label': 'Monthly expiry cycle levels', 
                'historical_levels': deque(maxlen=756),   # 3 years
                'level_formation_patterns': {},
                'dynamic_weights': {},
                'market_structure_adaptations': {}
            },
            'dte_31_plus': {
                'range': (31, 365),
                'label': 'Far month expiry levels',
                'historical_levels': deque(maxlen=504),    # 2 years
                'level_formation_patterns': {},
                'dynamic_weights': {},
                'market_structure_adaptations': {}
            }
        }
        
        # Dynamic Weight Learning Engine
        self.weight_learning_engine = SupportResistanceWeightLearner()
        
        # Historical Performance Tracker
        self.performance_tracker = LevelPerformanceTracker()
```

---

## Rolling Straddle Support/Resistance Detection

### Component 1 & 3 Straddle Levels Analysis

```python
def detect_straddle_support_resistance(self, straddle_data: dict, current_dte: int):
    """
    Detect support/resistance levels from rolling straddle prices
    Uses both Component 1 (ATM/ITM1/OTM1) and Component 3 (ATM±7 cumulative) data
    """
    
    straddle_levels = {}
    
    # Component 1: ATM/ITM1/OTM1 Straddle Levels
    component1_levels = self._detect_component1_levels(straddle_data['component_1'], current_dte)
    straddle_levels['component_1_levels'] = component1_levels
    
    # Component 3: Cumulative ATM±7 Straddle Levels  
    component3_levels = self._detect_component3_levels(straddle_data['component_3'], current_dte)
    straddle_levels['component_3_levels'] = component3_levels
    
    # Cross-validation between Component 1 and 3 levels
    validated_levels = self._cross_validate_straddle_levels(
        component1_levels, component3_levels, current_dte
    )
    
    return {
        'straddle_support_resistance': straddle_levels,
        'validated_levels': validated_levels,
        'level_confidence_scores': self._calculate_straddle_level_confidence(validated_levels),
        'dte_specific_adjustments': self._apply_dte_specific_adjustments(validated_levels, current_dte)
    }

def _detect_component1_levels(self, component1_data: dict, current_dte: int):
    """
    Detect support/resistance from Component 1 straddle prices (ATM/ITM1/OTM1)
    """
    
    levels = {}
    
    # ATM Straddle Levels (Highest Weight - 35%)
    atm_prices = component1_data['atm_straddle_prices']
    atm_levels = self._identify_price_levels(atm_prices, 'atm_straddle', current_dte)
    levels['atm_levels'] = {
        'support_levels': atm_levels['support'],
        'resistance_levels': atm_levels['resistance'],
        'weight': 0.35,
        'level_strength': self._calculate_level_strength(atm_levels, atm_prices)
    }
    
    # ITM1 Straddle Levels (20% Weight)
    itm1_prices = component1_data['itm1_straddle_prices']
    itm1_levels = self._identify_price_levels(itm1_prices, 'itm1_straddle', current_dte)
    levels['itm1_levels'] = {
        'support_levels': itm1_levels['support'],
        'resistance_levels': itm1_levels['resistance'],
        'weight': 0.20,
        'level_strength': self._calculate_level_strength(itm1_levels, itm1_prices)
    }
    
    # OTM1 Straddle Levels (20% Weight)
    otm1_prices = component1_data['otm1_straddle_prices']
    otm1_levels = self._identify_price_levels(otm1_prices, 'otm1_straddle', current_dte)
    levels['otm1_levels'] = {
        'support_levels': otm1_levels['support'],
        'resistance_levels': otm1_levels['resistance'],
        'weight': 0.20,
        'level_strength': self._calculate_level_strength(otm1_levels, otm1_prices)
    }
    
    # Apply learned weights from historical performance
    learned_weights = self._get_learned_component1_weights(current_dte)
    if learned_weights:
        levels = self._apply_learned_weights(levels, learned_weights)
    
    return levels

def _detect_component3_levels(self, component3_data: dict, current_dte: int):
    """
    Detect support/resistance from Component 3 cumulative ATM±7 straddle data
    """
    
    levels = {}
    
    # Cumulative CE Straddle Levels across ATM±7
    cumulative_ce_prices = component3_data['cumulative_ce_prices_atm_pm7']
    ce_levels = self._identify_price_levels(cumulative_ce_prices, 'cumulative_ce_atm_pm7', current_dte)
    
    # Cumulative PE Straddle Levels across ATM±7
    cumulative_pe_prices = component3_data['cumulative_pe_prices_atm_pm7']
    pe_levels = self._identify_price_levels(cumulative_pe_prices, 'cumulative_pe_atm_pm7', current_dte)
    
    # Combined Cumulative Straddle Levels
    combined_straddle_prices = component3_data['combined_cumulative_straddle_prices_atm_pm7']
    combined_levels = self._identify_price_levels(combined_straddle_prices, 'combined_cumulative_atm_pm7', current_dte)
    
    levels = {
        'cumulative_ce_levels': {
            'support_levels': ce_levels['support'],
            'resistance_levels': ce_levels['resistance'],
            'weight': 0.30,
            'level_strength': self._calculate_level_strength(ce_levels, cumulative_ce_prices)
        },
        'cumulative_pe_levels': {
            'support_levels': pe_levels['support'],
            'resistance_levels': pe_levels['resistance'],
            'weight': 0.30,
            'level_strength': self._calculate_level_strength(pe_levels, cumulative_pe_prices)
        },
        'combined_cumulative_levels': {
            'support_levels': combined_levels['support'],
            'resistance_levels': combined_levels['resistance'],
            'weight': 0.40,
            'level_strength': self._calculate_level_strength(combined_levels, combined_straddle_prices)
        }
    }
    
    # Apply learned weights from historical performance
    learned_weights = self._get_learned_component3_weights(current_dte)
    if learned_weights:
        levels = self._apply_learned_weights(levels, learned_weights)
    
    return levels

def _identify_price_levels(self, price_data: pd.Series, level_type: str, current_dte: int):
    """
    Core algorithm to identify support and resistance levels from price data
    Uses multiple techniques with historical learning
    """
    
    if len(price_data) < 50:
        return {'support': [], 'resistance': []}
    
    # Method 1: Pivot-based levels
    pivot_levels = self._detect_pivot_based_levels(price_data)
    
    # Method 2: Volume-based levels (if volume data available)
    volume_levels = self._detect_volume_based_levels(price_data, level_type)
    
    # Method 3: Psychological levels (round numbers)
    psychological_levels = self._detect_psychological_levels(price_data)
    
    # Method 4: Moving average confluence levels
    ma_confluence_levels = self._detect_ma_confluence_levels(price_data)
    
    # Method 5: Historical test levels
    historical_levels = self._detect_historical_test_levels(price_data, current_dte)
    
    # Combine all methods with learned weights
    method_weights = self._get_learned_method_weights(level_type, current_dte)
    
    combined_levels = self._combine_level_methods(
        [pivot_levels, volume_levels, psychological_levels, ma_confluence_levels, historical_levels],
        method_weights
    )
    
    # Filter and rank levels by strength
    filtered_levels = self._filter_and_rank_levels(combined_levels, price_data)
    
    return {
        'support': filtered_levels['support_levels'],
        'resistance': filtered_levels['resistance_levels'],
        'method_weights_used': method_weights,
        'level_quality_scores': filtered_levels['quality_scores']
    }
```

---

## Underlying Price Support/Resistance Detection

### Multi-Timeframe Underlying Analysis

```python
def detect_underlying_support_resistance(self, underlying_data: dict, current_dte: int):
    """
    Detect support/resistance levels from underlying prices across multiple timeframes
    Complements straddle-based levels for comprehensive analysis
    """
    
    underlying_levels = {}
    
    # Daily Timeframe Levels (40% Weight)
    daily_levels = self._detect_daily_underlying_levels(underlying_data['daily'], current_dte)
    underlying_levels['daily_levels'] = daily_levels
    
    # Weekly Timeframe Levels (35% Weight)
    weekly_levels = self._detect_weekly_underlying_levels(underlying_data['weekly'], current_dte)
    underlying_levels['weekly_levels'] = weekly_levels
    
    # Monthly Timeframe Levels (25% Weight)
    monthly_levels = self._detect_monthly_underlying_levels(underlying_data['monthly'], current_dte)
    underlying_levels['monthly_levels'] = monthly_levels
    
    # Cross-timeframe validation
    validated_underlying_levels = self._cross_validate_timeframe_levels(
        daily_levels, weekly_levels, monthly_levels, current_dte
    )
    
    return {
        'underlying_support_resistance': underlying_levels,
        'validated_underlying_levels': validated_underlying_levels,
        'timeframe_confluence_zones': self._identify_timeframe_confluence(underlying_levels),
        'underlying_level_strength': self._calculate_underlying_level_strength(validated_underlying_levels)
    }

def _detect_daily_underlying_levels(self, daily_data: dict, current_dte: int):
    """
    Detect daily timeframe support/resistance levels from underlying prices
    """
    
    daily_prices = daily_data['close_prices']
    daily_volumes = daily_data.get('volumes', None)
    
    # Daily-specific level detection methods
    methods = {
        'daily_pivots': self._detect_daily_pivots(daily_data),
        'previous_day_levels': self._detect_previous_day_levels(daily_data),
        'gap_levels': self._detect_gap_levels(daily_data),
        'volume_profile_levels': self._detect_volume_profile_levels(daily_data) if daily_volumes else {},
        'round_number_levels': self._detect_round_number_levels(daily_prices)
    }
    
    # Get learned weights for daily methods
    method_weights = self._get_learned_daily_method_weights(current_dte)
    
    # Combine methods
    combined_daily_levels = self._combine_daily_level_methods(methods, method_weights)
    
    return {
        'support_levels': combined_daily_levels['support'],
        'resistance_levels': combined_daily_levels['resistance'],
        'method_weights_used': method_weights,
        'daily_level_confidence': self._calculate_daily_level_confidence(combined_daily_levels)
    }

def _detect_weekly_underlying_levels(self, weekly_data: dict, current_dte: int):
    """
    Detect weekly timeframe support/resistance levels from underlying prices
    """
    
    weekly_prices = weekly_data['close_prices']
    
    # Weekly-specific level detection methods
    methods = {
        'weekly_pivots': self._detect_weekly_pivots(weekly_data),
        'weekly_high_low_levels': self._detect_weekly_high_low_levels(weekly_data),
        'weekly_open_levels': self._detect_weekly_open_levels(weekly_data),
        'weekly_ma_levels': self._detect_weekly_ma_levels(weekly_data)
    }
    
    # Get learned weights for weekly methods
    method_weights = self._get_learned_weekly_method_weights(current_dte)
    
    # Combine methods
    combined_weekly_levels = self._combine_weekly_level_methods(methods, method_weights)
    
    return {
        'support_levels': combined_weekly_levels['support'],
        'resistance_levels': combined_weekly_levels['resistance'],
        'method_weights_used': method_weights,
        'weekly_level_confidence': self._calculate_weekly_level_confidence(combined_weekly_levels)
    }

def _detect_monthly_underlying_levels(self, monthly_data: dict, current_dte: int):
    """
    Detect monthly timeframe support/resistance levels from underlying prices
    """
    
    monthly_prices = monthly_data['close_prices']
    
    # Monthly-specific level detection methods
    methods = {
        'monthly_pivots': self._detect_monthly_pivots(monthly_data),
        'monthly_high_low_levels': self._detect_monthly_high_low_levels(monthly_data),
        'long_term_ma_levels': self._detect_long_term_ma_levels(monthly_data),
        'fibonacci_retracement_levels': self._detect_fibonacci_levels(monthly_data)
    }
    
    # Get learned weights for monthly methods
    method_weights = self._get_learned_monthly_method_weights(current_dte)
    
    # Combine methods
    combined_monthly_levels = self._combine_monthly_level_methods(methods, method_weights)
    
    return {
        'support_levels': combined_monthly_levels['support'],
        'resistance_levels': combined_monthly_levels['resistance'],
        'method_weights_used': method_weights,
        'monthly_level_confidence': self._calculate_monthly_level_confidence(combined_monthly_levels)
    }
```

---

## Dynamic Weight Learning System

### Historical Performance-Based Weight Adjustment

```python
class SupportResistanceWeightLearner:
    """
    Advanced weight learning system for support/resistance detection
    Adapts weights based on historical performance across dual DTE framework
    """
    
    def __init__(self):
        # Learning configuration
        self.learning_config = {
            'minimum_samples': 50,        # Minimum trades for weight learning
            'lookback_periods': 252,      # 1 year of performance data
            'weight_adjustment_rate': 0.1, # 10% maximum adjustment per update
            'performance_metrics': ['accuracy', 'sharpe_ratio', 'max_drawdown', 'hit_ratio'],
            'method_performance_tracking': True
        }
        
        # Specific DTE Weight Learning
        self.specific_dte_weights = {}
        
        # DTE Range Weight Learning
        self.dte_range_weights = {}
        
        # Method-specific performance tracking
        self.method_performance = {}
    
    def learn_optimal_weights(self, performance_data: dict, level_type: str, current_dte: int):
        """
        Learn optimal weights for support/resistance detection methods
        Based on historical performance data
        """
        
        if len(performance_data) < self.learning_config['minimum_samples']:
            return self._get_default_weights(level_type)
        
        # Calculate performance metrics for each method
        method_performance = {}
        
        for method_name, method_results in performance_data.items():
            # Calculate comprehensive performance score
            performance_score = self._calculate_method_performance_score(method_results)
            method_performance[method_name] = performance_score
        
        # Convert performance scores to weights
        optimal_weights = self._convert_performance_to_weights(method_performance)
        
        # Store learned weights for specific DTE
        dte_key = f'dte_{current_dte}'
        if dte_key not in self.specific_dte_weights:
            self.specific_dte_weights[dte_key] = {}
        
        self.specific_dte_weights[dte_key][level_type] = optimal_weights
        
        # Also update DTE range weights
        dte_range = self._get_dte_range_category(current_dte)
        if dte_range not in self.dte_range_weights:
            self.dte_range_weights[dte_range] = {}
        
        self.dte_range_weights[dte_range][level_type] = optimal_weights
        
        return optimal_weights
    
    def get_adaptive_weights(self, level_type: str, current_dte: int):
        """
        Get adaptive weights for support/resistance detection
        Uses specific DTE weights if available, otherwise falls back to DTE range
        """
        
        # Try specific DTE weights first
        dte_key = f'dte_{current_dte}'
        if (dte_key in self.specific_dte_weights and 
            level_type in self.specific_dte_weights[dte_key]):
            return self.specific_dte_weights[dte_key][level_type]
        
        # Fall back to DTE range weights
        dte_range = self._get_dte_range_category(current_dte)
        if (dte_range in self.dte_range_weights and 
            level_type in self.dte_range_weights[dte_range]):
            return self.dte_range_weights[dte_range][level_type]
        
        # Final fallback to default weights
        return self._get_default_weights(level_type)
    
    def _calculate_method_performance_score(self, method_results: list):
        """
        Calculate comprehensive performance score for a detection method
        """
        
        if not method_results:
            return 0.5  # Neutral score
        
        # Extract performance metrics
        accuracies = [r.get('accuracy', 0.5) for r in method_results]
        sharpe_ratios = [r.get('sharpe_ratio', 0.0) for r in method_results]
        max_drawdowns = [r.get('max_drawdown', -0.1) for r in method_results]
        hit_ratios = [r.get('hit_ratio', 0.5) for r in method_results]
        
        # Calculate average metrics
        avg_accuracy = np.mean(accuracies)
        avg_sharpe = np.mean(sharpe_ratios)
        avg_max_dd = np.mean(max_drawdowns)
        avg_hit_ratio = np.mean(hit_ratios)
        
        # Composite performance score
        performance_score = (
            avg_accuracy * 0.35 +                    # Accuracy weight
            min(max(avg_sharpe / 2.0, 0), 1) * 0.25 + # Normalized Sharpe ratio
            (1 + avg_max_dd) * 0.25 +                # Drawdown (converted to positive)
            avg_hit_ratio * 0.15                     # Hit ratio weight
        )
        
        return max(0.1, min(1.0, performance_score))  # Bound between 0.1 and 1.0
    
    def _convert_performance_to_weights(self, method_performance: dict):
        """
        Convert method performance scores to normalized weights
        """
        
        total_performance = sum(method_performance.values())
        
        if total_performance == 0:
            # Equal weights if no performance data
            return {method: 1.0/len(method_performance) for method in method_performance}
        
        # Performance-proportional weights
        raw_weights = {}
        for method, performance in method_performance.items():
            raw_weights[method] = performance / total_performance
        
        # Apply constraints (minimum 5%, maximum 50%)
        constrained_weights = {}
        for method, weight in raw_weights.items():
            constrained_weights[method] = max(0.05, min(0.50, weight))
        
        # Renormalize
        total_weight = sum(constrained_weights.values())
        normalized_weights = {method: weight/total_weight for method, weight in constrained_weights.items()}
        
        return normalized_weights
    
    def update_performance_feedback(self, level_type: str, current_dte: int, 
                                  method_results: dict, actual_performance: dict):
        """
        Update method performance based on actual trading results
        """
        
        # Store performance feedback
        dte_key = f'dte_{current_dte}'
        
        if dte_key not in self.method_performance:
            self.method_performance[dte_key] = {}
        
        if level_type not in self.method_performance[dte_key]:
            self.method_performance[dte_key][level_type] = {}
        
        # Update each method's performance history
        for method_name, performance_data in actual_performance.items():
            if method_name not in self.method_performance[dte_key][level_type]:
                self.method_performance[dte_key][level_type][method_name] = deque(maxlen=252)
            
            self.method_performance[dte_key][level_type][method_name].append({
                'timestamp': datetime.now(),
                'performance_data': performance_data,
                'level_predictions': method_results.get(method_name, {}),
                'dte': current_dte
            })
        
        # Trigger weight relearning if enough new data
        recent_data_count = len(list(self.method_performance[dte_key][level_type].values())[0])
        if recent_data_count >= self.learning_config['minimum_samples']:
            self._trigger_weight_relearning(level_type, current_dte)
    
    def _trigger_weight_relearning(self, level_type: str, current_dte: int):
        """
        Trigger relearning of weights based on accumulated performance data
        """
        
        dte_key = f'dte_{current_dte}'
        
        if (dte_key in self.method_performance and 
            level_type in self.method_performance[dte_key]):
            
            # Extract recent performance data
            method_performance_data = self.method_performance[dte_key][level_type]
            
            # Convert to format suitable for learning
            learning_data = {}
            for method_name, performance_history in method_performance_data.items():
                learning_data[method_name] = list(performance_history)
            
            # Learn new optimal weights
            new_weights = self.learn_optimal_weights(learning_data, level_type, current_dte)
            
            logger.info(f"Updated weights for {level_type} DTE {current_dte}: {new_weights}")
    
    def _get_dte_range_category(self, dte: int) -> str:
        """Determine which DTE range category the current DTE belongs to"""
        if 0 <= dte <= 7:
            return 'dte_0_to_7'
        elif 8 <= dte <= 30:
            return 'dte_8_to_30'
        elif 31 <= dte <= 365:
            return 'dte_31_plus'
        else:
            return 'dte_0_to_7'  # Default fallback
    
    def _get_default_weights(self, level_type: str):
        """Get default weights when no learned data is available"""
        
        default_weights = {
            'straddle_levels': {
                'pivot_based': 0.30,
                'volume_based': 0.25,
                'psychological': 0.15,
                'ma_confluence': 0.20,
                'historical_test': 0.10
            },
            'underlying_daily': {
                'daily_pivots': 0.35,
                'previous_day_levels': 0.25,
                'gap_levels': 0.15,
                'volume_profile': 0.15,
                'round_numbers': 0.10
            },
            'underlying_weekly': {
                'weekly_pivots': 0.40,
                'weekly_high_low': 0.30,
                'weekly_open': 0.15,
                'weekly_ma': 0.15
            },
            'underlying_monthly': {
                'monthly_pivots': 0.35,
                'monthly_high_low': 0.25,
                'long_term_ma': 0.25,
                'fibonacci': 0.15
            }
        }
        
        return default_weights.get(level_type, {})
```

---

## Level Strength Assessment & Validation

### Multi-Factor Level Strength Calculation

```python
def calculate_comprehensive_level_strength(self, level_price: float, 
                                         price_history: pd.Series,
                                         volume_history: pd.Series,
                                         current_dte: int):
    """
    Calculate comprehensive strength score for support/resistance level
    Uses multiple validation factors with DTE-specific adjustments
    """
    
    strength_factors = {}
    
    # Factor 1: Historical Touch Count
    touch_count = self._count_level_touches(level_price, price_history)
    strength_factors['touch_count'] = min(1.0, touch_count / 5.0)  # Normalize to max 5 touches
    
    # Factor 2: Hold Success Rate
    hold_success_rate = self._calculate_hold_success_rate(level_price, price_history)
    strength_factors['hold_success'] = hold_success_rate
    
    # Factor 3: Volume Confirmation
    if volume_history is not None:
        volume_confirmation = self._calculate_volume_confirmation(level_price, price_history, volume_history)
        strength_factors['volume_confirmation'] = volume_confirmation
    else:
        strength_factors['volume_confirmation'] = 0.5  # Neutral if no volume data
    
    # Factor 4: Time-based Strength (age of level)
    time_strength = self._calculate_time_based_strength(level_price, price_history)
    strength_factors['time_strength'] = time_strength
    
    # Factor 5: Proximity to Round Numbers
    round_number_strength = self._calculate_round_number_proximity(level_price)
    strength_factors['round_number'] = round_number_strength
    
    # Factor 6: Multiple Timeframe Confirmation
    mtf_confirmation = self._calculate_mtf_confirmation(level_price, current_dte)
    strength_factors['mtf_confirmation'] = mtf_confirmation
    
    # Get DTE-specific factor weights
    factor_weights = self._get_dte_specific_strength_weights(current_dte)
    
    # Calculate weighted strength score
    weighted_strength = sum(
        strength_factors[factor] * factor_weights[factor] 
        for factor in strength_factors
    )
    
    return {
        'overall_strength': float(weighted_strength),
        'strength_factors': strength_factors,
        'factor_weights_used': factor_weights,
        'strength_classification': self._classify_level_strength(weighted_strength),
        'confidence_score': self._calculate_strength_confidence(strength_factors)
    }

def _get_dte_specific_strength_weights(self, current_dte: int):
    """
    Get DTE-specific weights for level strength factors
    Near expiry emphasizes different factors than far expiry
    """
    
    if current_dte <= 7:  # Weekly expiry cycle
        return {
            'touch_count': 0.25,        # High importance for recent tests
            'hold_success': 0.30,       # Critical for short-term levels
            'volume_confirmation': 0.20, # Volume patterns important
            'time_strength': 0.10,      # Less important for short-term
            'round_number': 0.10,       # Psychological levels matter
            'mtf_confirmation': 0.05    # Less relevant for short-term
        }
    elif current_dte <= 30:  # Monthly expiry cycle
        return {
            'touch_count': 0.20,        # Balanced approach
            'hold_success': 0.25,       # Important but not dominant
            'volume_confirmation': 0.20, # Standard volume analysis
            'time_strength': 0.15,      # Moderate time importance
            'round_number': 0.10,       # Standard psychological factor
            'mtf_confirmation': 0.10    # Some multi-timeframe value
        }
    else:  # Far month expiries (31+ days)
        return {
            'touch_count': 0.15,        # Less emphasis on recent tests
            'hold_success': 0.20,       # Moderate importance
            'volume_confirmation': 0.15, # Less volume emphasis
            'time_strength': 0.25,      # Higher time-based strength
            'round_number': 0.10,       # Standard psychological
            'mtf_confirmation': 0.15    # Higher multi-timeframe importance
        }
```

---

## Comprehensive Integration Framework

### Master Support/Resistance Analysis

```python
def analyze_comprehensive_support_resistance(self, all_data: dict, current_dte: int, market_context: dict):
    """
    Perform comprehensive support/resistance analysis with dual asset and dual DTE approach
    
    Args:
        all_data: Combined straddle and underlying price data
        current_dte: Current DTE for analysis
        market_context: Additional market context
        
    Returns:
        Comprehensive support/resistance analysis results
    """
    
    analysis_start = time.time()
    
    # Step 1: Rolling Straddle Support/Resistance Detection
    straddle_levels = self.detect_straddle_support_resistance(
        all_data['straddle_data'], current_dte
    )
    
    # Step 2: Underlying Price Support/Resistance Detection  
    underlying_levels = self.detect_underlying_support_resistance(
        all_data['underlying_data'], current_dte
    )
    
    # Step 3: Cross-Asset Level Validation
    cross_asset_validation = self._cross_validate_asset_levels(
        straddle_levels, underlying_levels, current_dte
    )
    
    # Step 4: Level Confluence Analysis
    confluence_analysis = self._analyze_level_confluence(
        straddle_levels, underlying_levels, current_dte
    )
    
    # Step 5: Dynamic Weight Application
    weighted_levels = self._apply_dynamic_weights(
        straddle_levels, underlying_levels, confluence_analysis, current_dte
    )
    
    # Step 6: Level Strength Assessment
    level_strength_analysis = self._assess_comprehensive_level_strength(
        weighted_levels, all_data, current_dte
    )
    
    # Step 7: Breakout/Breakdown Probability Calculation
    breakout_probability = self._calculate_breakout_probabilities(
        weighted_levels, level_strength_analysis, market_context, current_dte
    )
    
    # Step 8: Real-Time Level Monitoring Setup
    monitoring_setup = self._setup_real_time_level_monitoring(
        weighted_levels, current_dte
    )
    
    # Step 9: Historical Performance Update
    self._update_level_performance_tracking(
        weighted_levels, level_strength_analysis, current_dte
    )
    
    analysis_time = time.time() - analysis_start
    
    return {
        'timestamp': datetime.now().isoformat(),
        'component': 'Component 7: Support & Resistance Formation Logic',
        'dte': current_dte,
        'analysis_type': 'comprehensive_dual_asset_dual_dte',
        
        # Core Analysis Results
        'straddle_levels': straddle_levels,
        'underlying_levels': underlying_levels,
        'cross_asset_validation': cross_asset_validation,
        
        # Advanced Analysis
        'confluence_analysis': confluence_analysis,
        'weighted_levels': weighted_levels,
        'level_strength_analysis': level_strength_analysis,
        
        # Predictive Analysis
        'breakout_probability': breakout_probability,
        'key_levels_to_watch': self._identify_key_levels(weighted_levels, level_strength_analysis),
        
        # Monitoring & Learning
        'real_time_monitoring': monitoring_setup,
        'weight_learning_status': self._get_weight_learning_status(current_dte),
        
        # Performance Metrics
        'analysis_time_ms': analysis_time * 1000,
        'performance_target_met': analysis_time < 0.15,  # <150ms target
        
        # Component Health
        'component_health': {
            'straddle_level_engine_active': True,
            'underlying_level_engine_active': True,
            'weight_learning_engine_active': True,
            'dual_dte_engine_active': True,
            'historical_performance_tracking_active': True
        }
    }
```

---

## Performance Targets

### Component 7 Performance Requirements

```python
COMPONENT_7_PERFORMANCE_TARGETS = {
    'analysis_latency': {
        'comprehensive_analysis': '<150ms',
        'straddle_level_detection': '<60ms',
        'underlying_level_detection': '<50ms',
        'level_strength_assessment': '<40ms',
        'weight_learning_update': '<30ms'
    },
    
    'accuracy_targets': {
        'support_resistance_accuracy': '>88%',
        'breakout_prediction_accuracy': '>82%',
        'false_breakout_detection': '>85%',
        'level_strength_assessment_accuracy': '>90%'
    },
    
    'memory_usage': {
        'specific_dte_level_storage': '<200MB',  # For 91 specific DTEs
        'dte_range_level_storage': '<100MB',    # For 3 DTE ranges
        'historical_level_data': '<150MB',      # Historical levels and performance
        'weight_learning_data': '<100MB',       # Weight learning storage
        'real_time_monitoring': '<50MB',        # Level monitoring data
        'total_component_memory': '<600MB'
    },
    
    'learning_requirements': {
        'minimum_level_samples': 50,           # Minimum for weight learning
        'optimal_learning_depth': 252,        # 1 year of level performance data
        'weight_update_frequency': 'daily',   # Daily weight adjustments
        'cross_validation_accuracy': '>85%'   # Weight learning validation accuracy
    }
}
```

---

## Summary

Component 7 provides comprehensive **dynamic support and resistance detection** with advanced learning capabilities:

### Key Features:
1. **Dual Asset Analysis**: Both rolling straddle prices AND underlying prices
2. **Multi-Straddle Integration**: Component 1 (ATM/ITM1/OTM1) + Component 3 (ATM±7 cumulative)
3. **Historical Weight Learning**: All detection methods adapt based on performance feedback
4. **Dual DTE Framework**: Specific DTE and DTE range-based level analysis
5. **Real-Time Monitoring**: Continuous level strength assessment and breakout prediction

### Level Detection Sources:
- **Straddle Levels**: ATM, ITM1, OTM1, Cumulative ATM±7 CE/PE levels
- **Underlying Levels**: Daily, weekly, monthly timeframe levels
- **Cross-Asset Validation**: Straddle vs underlying level confirmation
- **Multiple Detection Methods**: Pivots, volume, psychological, MA confluence, historical tests

### Dynamic Learning:
- **Method Weights**: Each detection method weight adjusts based on historical accuracy
- **DTE-Specific Learning**: Different weights for expiry proximity (dte=0 vs dte=30)
- **Performance Feedback**: Real trading results improve future level detection
- **Cross-Validation**: Level quality validated across multiple timeframes and assets

**Performance Optimized:**
- <150ms comprehensive analysis
- <600MB total memory usage
- >88% support/resistance accuracy
- Real-time level monitoring with breakout prediction

Component 7 completes the advanced level detection foundation, providing dynamic support/resistance analysis that adapts to any market structure through continuous learning.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Enhance Component 2 with historical learning for weights, sentiment thresholds, and volume thresholds", "status": "completed", "id": "2"}, {"content": "Add detailed logic section explaining the Greeks sentiment system", "status": "completed", "id": "2b"}, {"content": "Create Component 3: OI-PA Trending Analysis", "status": "completed", "id": "3"}, {"content": "Enhance Component 3 with expert recommendations", "status": "completed", "id": "3b"}, {"content": "Create Component 4: IV Skew Analysis", "status": "completed", "id": "4"}, {"content": "Create Component 5: ATR-EMA with CPR Integration", "status": "completed", "id": "5"}, {"content": "Create Component 6: Correlation & Non-Correlation Framework", "status": "completed", "id": "6"}, {"content": "Create Component 7: Support & Resistance Formation Logic", "status": "completed", "id": "7"}, {"content": "Create Component 8: DTE-Adaptive Overlay System", "status": "in_progress", "id": "8"}, {"content": "Create Master Document mr_master_v1.md", "status": "pending", "id": "9"}]