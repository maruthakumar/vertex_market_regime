# Component 4: IV Skew Analysis System

> Vertex AI Feature Engineering (Required): 87 features to be engineered by Vertex AI Pipelines and managed via Vertex AI Feature Store with training/serving parity. Data: GCS Parquet → Arrow/RAPIDS.

## Market Regime Classification Framework

---

## **System Logic & Methodology**

### **Core Concept**
The IV Skew Analysis System represents a sophisticated approach to understanding market sentiment and directional bias through implied volatility patterns across strike prices. This system recognizes that IV skew patterns reveal institutional positioning, market fear/greed levels, and upcoming directional moves that precede significant price movements.

### **Revolutionary Multi-Strike IV Skew Approach**
Unlike traditional IV analysis that focuses on single strikes or ATM volatility, our system analyzes **comprehensive IV skew patterns across ATM ±7 strikes** with adaptive learning to identify:
- **Institutional Directional Positioning** - Detected through asymmetric IV patterns
- **Market Regime Transition Signals** - Identified via skew flattening/steepening cycles
- **Tail Risk Assessment** - Measured through wing volatility premiums
- **Smart Money Flow Detection** - Revealed through unusual skew deviations

### **Critical Methodology: Comprehensive IV Skew Analysis**
- **Strike Range**: ATM ±7 strikes (consistent with Component 3 OI analysis)
- **Skew Metrics**: Put skew, Call skew, Term structure analysis, Volatility smile analysis
- **Adaptive Learning**: DTE-specific and all-days learning for skew parameters
- **Multi-Timeframe Integration**: 5min, 15min, and intraday skew evolution analysis
- **Symbol-Specific Calibration**: NIFTY/BANKNIFTY/Stocks optimized parameters

### **Key Innovation**
The system combines **real-time IV skew analysis** with **historical pattern recognition** and **adaptive parameter learning** to identify regime changes before they become apparent in price action, providing significant alpha generation opportunities.

---

## **1. IV Skew Calculation & Analysis Engine**

### **Multi-Strike IV Skew Calculator**
```python
class IVSkewAnalysisEngine:
    def __init__(self):
        # IV Skew calculation parameters
        self.skew_config = {
            'strike_range': 7,              # ATM ±7 strikes (consistent with OI analysis)
            'skew_calculation_methods': ['polynomial', 'linear', 'smile_fitting'],
            'outlier_threshold': 3.0,       # Standard deviations for outlier detection
            'min_strikes_required': 5,      # Minimum strikes needed for valid skew
            'volatility_floor': 0.05,       # 5% minimum IV floor
            'volatility_ceiling': 2.0       # 200% maximum IV ceiling
        }
        
        # Symbol-specific IV characteristics (learned from historical data)
        self.symbol_iv_characteristics = {
            'NIFTY': {
                'typical_atm_iv': 0.18,      # 18% typical ATM IV
                'put_skew_normal': -0.05,    # Normal put skew
                'call_skew_normal': 0.02,    # Normal call skew
                'skew_volatility_factor': 1.0
            },
            'BANKNIFTY': {
                'typical_atm_iv': 0.22,      # 22% typical ATM IV
                'put_skew_normal': -0.08,    # Higher put skew
                'call_skew_normal': 0.03,    # Higher call skew
                'skew_volatility_factor': 1.3
            },
            'STOCKS': {
                'typical_atm_iv': 0.35,      # 35% typical ATM IV
                'put_skew_normal': -0.12,    # Much higher put skew
                'call_skew_normal': 0.05,    # Higher call skew
                'skew_volatility_factor': 1.8
            }
        }
        
        # DTE-specific skew adjustments
        self.dte_skew_adjustments = {
            0: {'put_multiplier': 1.8, 'call_multiplier': 1.5},    # Expiry day extremes
            1: {'put_multiplier': 1.4, 'call_multiplier': 1.2},    # Day before expiry
            2: {'put_multiplier': 1.2, 'call_multiplier': 1.1},    # 2 days to expiry
            3: {'put_multiplier': 1.1, 'call_multiplier': 1.05},   # 3 days to expiry
            'default': {'put_multiplier': 1.0, 'call_multiplier': 1.0}  # Normal conditions
        }
    
    def calculate_comprehensive_iv_skew(self, option_chain_data, underlying_price, current_dte, symbol_type):
        """Calculate comprehensive IV skew across ATM ±7 strikes"""
        skew_results = {}
        
        # 1. Extract relevant strikes (ATM ±7)
        relevant_strikes = self.extract_relevant_strikes_for_iv(
            option_chain_data, underlying_price, symbol_type
        )
        
        # 2. Calculate basic skew metrics
        basic_skew_metrics = self.calculate_basic_skew_metrics(relevant_strikes, underlying_price)
        
        # 3. Advanced skew analysis
        advanced_skew_analysis = self.calculate_advanced_skew_analysis(
            relevant_strikes, underlying_price, current_dte, symbol_type
        )
        
        # 4. Term structure analysis
        term_structure_analysis = self.calculate_iv_term_structure(
            relevant_strikes, current_dte, symbol_type
        )
        
        # 5. Skew anomaly detection
        skew_anomalies = self.detect_skew_anomalies(
            basic_skew_metrics, advanced_skew_analysis, symbol_type
        )
        
        skew_results = {
            'relevant_strikes': relevant_strikes,
            'basic_skew_metrics': basic_skew_metrics,
            'advanced_skew_analysis': advanced_skew_analysis,
            'term_structure_analysis': term_structure_analysis,
            'skew_anomalies': skew_anomalies,
            'overall_skew_assessment': self.generate_overall_skew_assessment(
                basic_skew_metrics, advanced_skew_analysis, skew_anomalies
            )
        }
        
        return skew_results
    
    def extract_relevant_strikes_for_iv(self, option_chain_data, underlying_price, symbol_type):
        """Extract IV data for ATM ±7 strikes"""
        # Determine strike interval based on symbol
        strike_intervals = {'NIFTY': 50, 'BANKNIFTY': 100, 'STOCKS': 25}
        strike_interval = strike_intervals.get(symbol_type, 50)
        
        # Calculate ATM and boundaries
        atm_strike = round(underlying_price / strike_interval) * strike_interval
        lower_bound = atm_strike - (7 * strike_interval)
        upper_bound = atm_strike + (7 * strike_interval)
        
        # Extract relevant strikes with IV data
        relevant_strikes = {
            'call_strikes': {},
            'put_strikes': {},
            'atm_strike': atm_strike,
            'strike_range': {'lower': lower_bound, 'upper': upper_bound}
        }
        
        # Process call options
        for strike_str, option_data in option_chain_data.get('CE', {}).items():
            strike = float(strike_str)
            if lower_bound <= strike <= upper_bound and option_data.get('IV') is not None:
                relevant_strikes['call_strikes'][strike] = {
                    'iv': option_data.get('IV', 0),
                    'price': option_data.get('LTP', 0),
                    'volume': option_data.get('volume', 0),
                    'oi': option_data.get('OI', 0),
                    'moneyness': (strike - underlying_price) / underlying_price
                }
        
        # Process put options
        for strike_str, option_data in option_chain_data.get('PE', {}).items():
            strike = float(strike_str)
            if lower_bound <= strike <= upper_bound and option_data.get('IV') is not None:
                relevant_strikes['put_strikes'][strike] = {
                    'iv': option_data.get('IV', 0),
                    'price': option_data.get('LTP', 0),
                    'volume': option_data.get('volume', 0),
                    'oi': option_data.get('OI', 0),
                    'moneyness': (strike - underlying_price) / underlying_price
                }
        
        return relevant_strikes
    
    def calculate_basic_skew_metrics(self, relevant_strikes, underlying_price):
        """Calculate fundamental IV skew metrics"""
        basic_metrics = {}
        
        call_strikes = relevant_strikes['call_strikes']
        put_strikes = relevant_strikes['put_strikes']
        atm_strike = relevant_strikes['atm_strike']
        
        # ATM IV (interpolated if necessary)
        atm_iv = self.interpolate_atm_iv(call_strikes, put_strikes, atm_strike)
        
        # Put skew calculation (OTM puts vs ATM)
        otm_put_strikes = {k: v for k, v in put_strikes.items() if k < underlying_price}
        if otm_put_strikes:
            # 10-delta put equivalent (approximately 1 strike away)
            put_10delta_iv = self.get_approximate_delta_iv(otm_put_strikes, 0.10, 'put')
            put_skew = put_10delta_iv - atm_iv
        else:
            put_skew = 0.0
        
        # Call skew calculation (OTM calls vs ATM) 
        otm_call_strikes = {k: v for k, v in call_strikes.items() if k > underlying_price}
        if otm_call_strikes:
            # 10-delta call equivalent
            call_10delta_iv = self.get_approximate_delta_iv(otm_call_strikes, 0.10, 'call')
            call_skew = call_10delta_iv - atm_iv
        else:
            call_skew = 0.0
        
        # Overall skew (put skew - call skew)
        overall_skew = put_skew - call_skew
        
        # Skew ratio
        skew_ratio = abs(put_skew) / (abs(call_skew) + 1e-6)
        
        basic_metrics = {
            'atm_iv': atm_iv,
            'put_skew': put_skew,
            'call_skew': call_skew,
            'overall_skew': overall_skew,
            'skew_ratio': skew_ratio,
            'put_wing_iv': put_10delta_iv if 'put_10delta_iv' in locals() else atm_iv,
            'call_wing_iv': call_10delta_iv if 'call_10delta_iv' in locals() else atm_iv
        }
        
        return basic_metrics
    
    def interpolate_atm_iv(self, call_strikes, put_strikes, atm_strike):
        """Interpolate ATM IV from available strikes"""
        # Try to get exact ATM IV
        if atm_strike in call_strikes and atm_strike in put_strikes:
            return (call_strikes[atm_strike]['iv'] + put_strikes[atm_strike]['iv']) / 2
        
        # Interpolate from nearby strikes
        all_strikes_with_iv = []
        
        # Combine call and put strikes near ATM
        for strike, data in call_strikes.items():
            if abs(strike - atm_strike) <= 100:  # Within reasonable range
                all_strikes_with_iv.append((strike, data['iv']))
        
        for strike, data in put_strikes.items():
            if abs(strike - atm_strike) <= 100:
                all_strikes_with_iv.append((strike, data['iv']))
        
        if len(all_strikes_with_iv) >= 2:
            # Simple linear interpolation
            all_strikes_with_iv.sort(key=lambda x: abs(x[0] - atm_strike))
            closest_strikes = all_strikes_with_iv[:2]
            
            if len(closest_strikes) == 2:
                s1, iv1 = closest_strikes[0]
                s2, iv2 = closest_strikes[1]
                
                # Linear interpolation
                weight = abs(s2 - atm_strike) / (abs(s2 - s1) + 1e-6)
                interpolated_iv = iv1 * weight + iv2 * (1 - weight)
                return interpolated_iv
            else:
                return closest_strikes[0][1]
        
        # Fallback to any available IV
        if call_strikes:
            return list(call_strikes.values())[0]['iv']
        elif put_strikes:
            return list(put_strikes.values())[0]['iv']
        else:
            return 0.20  # Default IV if nothing available
```

---

## **2. Advanced Skew Pattern Recognition**

### **Skew Pattern Classification Engine**
```python
class SkewPatternClassificationEngine:
    def __init__(self):
        # Skew pattern thresholds (learned from historical data)
        self.pattern_thresholds = {
            'normal_skew': {'put_skew_range': (-0.08, -0.02), 'call_skew_range': (0.01, 0.05)},
            'steep_put_skew': {'put_skew_threshold': -0.12, 'significance': 'bearish'},
            'flat_skew': {'overall_skew_threshold': 0.03, 'significance': 'neutral'},
            'reverse_skew': {'call_skew_higher': True, 'significance': 'very_bullish'},
            'volatility_smile': {'symmetric_threshold': 0.85, 'significance': 'uncertainty'},
            'term_structure_inversion': {'front_higher_than_back': True, 'significance': 'event_risk'}
        }
        
        # Pattern confidence scoring
        self.pattern_confidence_weights = {
            'volume_confirmation': 0.25,    # High volume in unusual strikes
            'oi_confirmation': 0.20,        # High OI supporting pattern
            'historical_deviation': 0.30,   # Deviation from normal patterns
            'cross_validation': 0.25        # Multiple timeframe confirmation
        }
    
    def classify_skew_patterns(self, skew_data, historical_patterns, symbol_type):
        """Classify IV skew patterns and their market implications"""
        pattern_analysis = {}
        
        basic_metrics = skew_data['basic_skew_metrics']
        advanced_analysis = skew_data['advanced_skew_analysis']
        
        # 1. Primary Pattern Classification
        primary_patterns = self.identify_primary_skew_patterns(basic_metrics, symbol_type)
        
        # 2. Secondary Pattern Features
        secondary_features = self.identify_secondary_pattern_features(advanced_analysis)
        
        # 3. Historical Context Analysis
        historical_context = self.analyze_historical_skew_context(
            basic_metrics, historical_patterns, symbol_type
        )
        
        # 4. Pattern Confidence Scoring
        pattern_confidence = self.calculate_pattern_confidence(
            primary_patterns, secondary_features, skew_data['relevant_strikes']
        )
        
        # 5. Market Implication Assessment
        market_implications = self.assess_market_implications(
            primary_patterns, secondary_features, historical_context, symbol_type
        )
        
        pattern_analysis = {
            'primary_patterns': primary_patterns,
            'secondary_features': secondary_features,
            'historical_context': historical_context,
            'pattern_confidence': pattern_confidence,
            'market_implications': market_implications,
            'overall_assessment': self.generate_overall_pattern_assessment(
                primary_patterns, market_implications, pattern_confidence
            )
        }
        
        return pattern_analysis
    
    def identify_primary_skew_patterns(self, basic_metrics, symbol_type):
        """Identify primary IV skew patterns"""
        patterns = {}
        
        put_skew = basic_metrics['put_skew']
        call_skew = basic_metrics['call_skew']
        overall_skew = basic_metrics['overall_skew']
        skew_ratio = basic_metrics['skew_ratio']
        
        # Get symbol-specific normal ranges
        symbol_chars = self.get_symbol_characteristics(symbol_type)
        
        # Pattern 1: Normal Skew
        if (symbol_chars['put_skew_normal'] * 0.8 <= put_skew <= symbol_chars['put_skew_normal'] * 1.2 and
            symbol_chars['call_skew_normal'] * 0.8 <= call_skew <= symbol_chars['call_skew_normal'] * 1.2):
            patterns['normal_skew'] = {
                'detected': True,
                'confidence': 0.8,
                'implication': 'neutral'
            }
        
        # Pattern 2: Steep Put Skew (Fear)
        if put_skew < symbol_chars['put_skew_normal'] * 1.5:
            patterns['steep_put_skew'] = {
                'detected': True,
                'confidence': min(abs(put_skew / symbol_chars['put_skew_normal']), 1.0),
                'implication': 'bearish_fear'
            }
        
        # Pattern 3: Flat Skew (Complacency)
        if abs(overall_skew) < abs(symbol_chars['put_skew_normal']) * 0.3:
            patterns['flat_skew'] = {
                'detected': True,
                'confidence': 0.7,
                'implication': 'complacency'
            }
        
        # Pattern 4: Reverse Skew (Extreme Bullishness)
        if call_skew > abs(put_skew):
            patterns['reverse_skew'] = {
                'detected': True,
                'confidence': min(call_skew / abs(put_skew), 1.0) if put_skew != 0 else 0.5,
                'implication': 'extreme_bullish'
            }
        
        # Pattern 5: Volatility Smile (Uncertainty)
        if (abs(put_skew) > abs(symbol_chars['put_skew_normal']) and 
            call_skew > symbol_chars['call_skew_normal'] and
            skew_ratio > 0.7 and skew_ratio < 1.3):
            patterns['volatility_smile'] = {
                'detected': True,
                'confidence': 0.8,
                'implication': 'high_uncertainty'
            }
        
        return patterns
    
    def get_symbol_characteristics(self, symbol_type):
        """Get symbol-specific IV characteristics"""
        if symbol_type.startswith('NIFTY'):
            return {
                'put_skew_normal': -0.05,
                'call_skew_normal': 0.02,
                'typical_atm_iv': 0.18
            }
        elif symbol_type.startswith('BANKNIFTY'):
            return {
                'put_skew_normal': -0.08,
                'call_skew_normal': 0.03,
                'typical_atm_iv': 0.22
            }
        else:  # STOCKS
            return {
                'put_skew_normal': -0.12,
                'call_skew_normal': 0.05,
                'typical_atm_iv': 0.35
            }
```

---

## **3. IV Skew Regime Classification**

### **Skew-Based Regime Detection Engine**
```python
class IVSkewRegimeDetectionEngine:
    def __init__(self):
        # IV Skew regime classification parameters
        self.regime_classification_params = {
            'regime_thresholds': {
                'LVLD': {'skew_range': (-0.06, 0.03), 'atm_iv_max': 0.20},  # Low Vol Low Divergence
                'HVC': {'steep_put_skew': True, 'atm_iv_rising': True},       # High Vol Convergence
                'VCPE': {'smile_pattern': True, 'wing_premiums_high': True},   # Vol Convergence Price Expansion
                'TBVE': {'skew_flattening': True, 'atm_iv_spiking': True},    # Trend Break Vol Expansion
                'TBVS': {'reverse_skew_signal': True, 'vol_compression': True}, # Trend Break Vol Squeeze
                'SCGS': {'symmetric_smile': True, 'gamma_squeeze': True},      # Strong Correlation Gamma Squeeze
                'PSED': {'term_structure_inversion': True, 'put_skew_extreme': True}, # Price Squeeze Expansion Divergence
                'CBV': {'skew_breakdown': True, 'correlation_break': True}     # Correlation Break Velocity
            },
            
            'regime_transition_signals': {
                'skew_velocity': 0.02,        # Rate of skew change threshold
                'atm_iv_velocity': 0.05,      # Rate of ATM IV change threshold
                'wing_spread_change': 0.03,   # Wing IV spread change threshold
                'term_structure_slope': 0.10  # Term structure slope change threshold
            }
        }
        
        # Historical regime pattern learning
        self.historical_regime_patterns = {
            'regime_duration_avg': {},      # Average duration in each regime
            'transition_probabilities': {}, # Probability of regime transitions
            'skew_pattern_persistence': {}, # How long skew patterns persist
            'regime_accuracy_scores': {}    # Historical accuracy of regime calls
        }
    
    def detect_iv_skew_regime(self, skew_analysis, market_data, current_regime, symbol_type):
        """Detect market regime based on IV skew analysis"""
        regime_detection = {}
        
        # Extract key skew metrics
        pattern_analysis = skew_analysis['pattern_analysis']
        basic_metrics = skew_analysis['basic_skew_metrics']
        term_structure = skew_analysis['term_structure_analysis']
        
        # 1. Calculate regime scores for each regime type
        regime_scores = self.calculate_regime_scores(
            pattern_analysis, basic_metrics, term_structure, market_data, symbol_type
        )
        
        # 2. Apply regime transition logic
        transition_analysis = self.analyze_regime_transitions(
            regime_scores, current_regime, symbol_type
        )
        
        # 3. Validate regime detection with multiple signals
        regime_validation = self.validate_regime_detection(
            regime_scores, transition_analysis, skew_analysis
        )
        
        # 4. Generate final regime assessment
        final_regime_assessment = self.generate_final_regime_assessment(
            regime_scores, transition_analysis, regime_validation
        )
        
        regime_detection = {
            'regime_scores': regime_scores,
            'transition_analysis': transition_analysis,
            'regime_validation': regime_validation,
            'final_assessment': final_regime_assessment,
            'confidence_level': self.calculate_regime_confidence(final_regime_assessment, regime_validation)
        }
        
        return regime_detection
    
    def calculate_regime_scores(self, pattern_analysis, basic_metrics, term_structure, market_data, symbol_type):
        """Calculate scores for each potential regime"""
        regime_scores = {}
        
        # Extract key metrics
        put_skew = basic_metrics['put_skew']
        call_skew = basic_metrics['call_skew']
        overall_skew = basic_metrics['overall_skew']
        atm_iv = basic_metrics['atm_iv']
        
        # Get symbol-specific parameters
        symbol_chars = self.get_symbol_characteristics(symbol_type)
        
        # LVLD - Low Volatility Low Divergence
        regime_scores['LVLD'] = self.calculate_lvld_score(
            atm_iv, overall_skew, symbol_chars, pattern_analysis
        )
        
        # HVC - High Volatility Convergence  
        regime_scores['HVC'] = self.calculate_hvc_score(
            put_skew, atm_iv, symbol_chars, pattern_analysis
        )
        
        # VCPE - Volatility Convergence Price Expansion
        regime_scores['VCPE'] = self.calculate_vcpe_score(
            pattern_analysis, basic_metrics, symbol_chars
        )
        
        # TBVE - Trend Break Volatility Expansion
        regime_scores['TBVE'] = self.calculate_tbve_score(
            basic_metrics, term_structure, market_data, pattern_analysis
        )
        
        # TBVS - Trend Break Volatility Squeeze
        regime_scores['TBVS'] = self.calculate_tbvs_score(
            call_skew, put_skew, atm_iv, pattern_analysis, symbol_chars
        )
        
        # SCGS - Strong Correlation Gamma Squeeze
        regime_scores['SCGS'] = self.calculate_scgs_score(
            pattern_analysis, basic_metrics, market_data
        )
        
        # PSED - Price Squeeze Expansion Divergence
        regime_scores['PSED'] = self.calculate_psed_score(
            term_structure, put_skew, pattern_analysis, symbol_chars
        )
        
        # CBV - Correlation Break Velocity
        regime_scores['CBV'] = self.calculate_cbv_score(
            basic_metrics, pattern_analysis, market_data
        )
        
        return regime_scores
    
    def calculate_lvld_score(self, atm_iv, overall_skew, symbol_chars, pattern_analysis):
        """Calculate LVLD regime score"""
        score = 0.0
        
        # Low ATM IV component
        if atm_iv < symbol_chars['typical_atm_iv'] * 0.9:
            score += 0.3 * (1 - atm_iv / symbol_chars['typical_atm_iv'])
        
        # Low overall skew component
        if abs(overall_skew) < abs(symbol_chars['put_skew_normal']) * 0.6:
            score += 0.4 * (1 - abs(overall_skew) / abs(symbol_chars['put_skew_normal']))
        
        # Normal pattern confirmation
        if pattern_analysis['primary_patterns'].get('normal_skew', {}).get('detected', False):
            score += 0.3 * pattern_analysis['primary_patterns']['normal_skew']['confidence']
        
        return min(score, 1.0)
    
    def calculate_hvc_score(self, put_skew, atm_iv, symbol_chars, pattern_analysis):
        """Calculate HVC regime score"""
        score = 0.0
        
        # Steep put skew component
        if put_skew < symbol_chars['put_skew_normal'] * 1.3:
            score += 0.4 * abs(put_skew / symbol_chars['put_skew_normal'])
        
        # Rising ATM IV component
        if atm_iv > symbol_chars['typical_atm_iv'] * 1.1:
            score += 0.3 * (atm_iv / symbol_chars['typical_atm_iv'] - 1)
        
        # Steep put skew pattern confirmation
        if pattern_analysis['primary_patterns'].get('steep_put_skew', {}).get('detected', False):
            score += 0.3 * pattern_analysis['primary_patterns']['steep_put_skew']['confidence']
        
        return min(score, 1.0)
    
    def calculate_vcpe_score(self, pattern_analysis, basic_metrics, symbol_chars):
        """Calculate VCPE regime score"""
        score = 0.0
        
        # Volatility smile pattern
        if pattern_analysis['primary_patterns'].get('volatility_smile', {}).get('detected', False):
            score += 0.5 * pattern_analysis['primary_patterns']['volatility_smile']['confidence']
        
        # High wing premiums
        wing_premium_avg = (basic_metrics['put_wing_iv'] + basic_metrics['call_wing_iv']) / 2
        if wing_premium_avg > basic_metrics['atm_iv'] * 1.2:
            score += 0.3 * ((wing_premium_avg / basic_metrics['atm_iv']) - 1)
        
        # Elevated overall volatility
        if basic_metrics['atm_iv'] > symbol_chars['typical_atm_iv'] * 1.15:
            score += 0.2
        
        return min(score, 1.0)
```

---

## **4. Adaptive Learning for IV Parameters**

### **Dual DTE Analysis Framework for IV Skew**
```python
class DualDTEIVSkewAnalysisEngine:
    def __init__(self):
        # Specific DTE Analysis (dte=0, dte=1, dte=7, dte=30, etc.)
        self.specific_dte_configs = {
            f'dte_{i}': {
                'historical_iv_data': deque(maxlen=252),  # 1 year of specific DTE data
                'skew_percentiles': {},
                'learned_parameters': {},
                'analysis_count': 0,
                'regime_accuracy': 0.0
            } for i in range(91)  # DTE 0 to 90
        }
        
        # DTE Range Analysis  
        self.dte_range_configs = {
            'dte_0_to_7': {
                'range': (0, 7),
                'label': 'Weekly expiry cycle',
                'historical_iv_data': deque(maxlen=1260),  # 5 years of weekly data
                'skew_percentiles': {},
                'learned_parameters': {},
                'weight_factor': 1.2  # Higher weight for short-term analysis
            },
            'dte_8_to_30': {
                'range': (8, 30),
                'label': 'Monthly expiry cycle',
                'historical_iv_data': deque(maxlen=756),  # 3 years of monthly data
                'skew_percentiles': {},
                'learned_parameters': {},
                'weight_factor': 1.0  # Standard weight
            },
            'dte_31_plus': {
                'range': (31, 365),
                'label': 'Far month expiries',
                'historical_iv_data': deque(maxlen=504),  # 2 years of far month data
                'skew_percentiles': {},
                'learned_parameters': {},
                'weight_factor': 0.8  # Lower weight for long-term
            }
        }

    def analyze_dual_dte_iv_skew(self, option_chain_data, underlying_price, current_dte, symbol_type):
        """Comprehensive IV skew analysis with dual DTE approach"""
        dual_analysis_results = {}
        
        # Step 1: Specific DTE Analysis
        specific_dte_results = self.analyze_specific_dte_iv_skew(
            option_chain_data, underlying_price, current_dte, symbol_type
        )
        
        # Step 2: DTE Range Analysis
        dte_range_results = self.analyze_dte_range_iv_skew(
            option_chain_data, underlying_price, current_dte, symbol_type
        )
        
        # Step 3: Cross-validate both approaches
        cross_validation = self.cross_validate_dte_approaches(
            specific_dte_results, dte_range_results, current_dte
        )
        
        # Step 4: Generate blended analysis
        blended_analysis = self.blend_dte_analysis_approaches(
            specific_dte_results, dte_range_results, cross_validation
        )
        
        dual_analysis_results = {
            'current_dte': current_dte,
            'analysis_type': 'dual_dte_iv_skew',
            'specific_dte_analysis': specific_dte_results,
            'dte_range_analysis': dte_range_results,
            'cross_validation': cross_validation,
            'blended_analysis': blended_analysis,
            'recommended_approach': self.recommend_analysis_approach(cross_validation)
        }
        
        return dual_analysis_results

    def analyze_specific_dte_iv_skew(self, option_chain_data, underlying_price, current_dte, symbol_type):
        """Analyze IV skew percentiles for specific DTE (e.g., dte=0, dte=7, dte=30)"""
        dte_key = f'dte_{current_dte}'
        
        if dte_key not in self.specific_dte_configs:
            return self._initialize_new_specific_dte(current_dte)
        
        dte_config = self.specific_dte_configs[dte_key]
        historical_data = dte_config['historical_iv_data']
        
        # Calculate current IV skew metrics
        current_skew_analysis = self.calculate_comprehensive_iv_skew(
            option_chain_data, underlying_price, current_dte, symbol_type
        )
        
        # Store current data point for this specific DTE
        current_data_point = {
            'timestamp': datetime.now(),
            'dte': current_dte,
            'put_skew': current_skew_analysis['basic_skew_metrics']['put_skew'],
            'call_skew': current_skew_analysis['basic_skew_metrics']['call_skew'],
            'overall_skew': current_skew_analysis['basic_skew_metrics']['overall_skew'],
            'atm_iv': current_skew_analysis['basic_skew_metrics']['atm_iv'],
            'skew_ratio': current_skew_analysis['basic_skew_metrics']['skew_ratio'],
            'pattern_detected': current_skew_analysis.get('pattern_analysis', {}).get('primary_patterns', {}),
            'regime_implications': current_skew_analysis.get('regime_detection', {})
        }
        
        historical_data.append(current_data_point)
        
        if len(historical_data) >= 30:  # Minimum data for percentile analysis
            # Calculate specific DTE percentiles
            specific_percentiles = self._calculate_specific_dte_percentiles(
                historical_data, current_dte
            )
            
            # Learn parameters for this specific DTE
            learned_params = self._learn_specific_dte_parameters(
                historical_data, current_dte
            )
            
            # Classify regime based on specific DTE patterns
            regime_classification = self._classify_specific_dte_regime(
                current_data_point, specific_percentiles, learned_params
            )
            
            dte_config['skew_percentiles'] = specific_percentiles
            dte_config['learned_parameters'] = learned_params
            dte_config['analysis_count'] += 1
            
            return {
                'dte': current_dte,
                'analysis_type': 'specific_dte_iv_skew',
                'percentiles': specific_percentiles,
                'learned_parameters': learned_params,
                'current_metrics': current_data_point,
                'regime_classification': regime_classification,
                'confidence': self._calculate_specific_dte_confidence(
                    current_data_point, specific_percentiles, len(historical_data)
                ),
                'data_quality': {
                    'historical_points': len(historical_data),
                    'analysis_count': dte_config['analysis_count'],
                    'data_sufficiency': 'sufficient' if len(historical_data) >= 50 else 'minimum'
                }
            }
        
        return {
            'dte': current_dte,
            'analysis_type': 'specific_dte_iv_skew',
            'status': 'insufficient_data',
            'data_points': len(historical_data),
            'required_minimum': 30
        }

    def analyze_dte_range_iv_skew(self, option_chain_data, underlying_price, current_dte, symbol_type):
        """Analyze IV skew percentiles for DTE ranges (weekly/monthly/far month)"""
        dte_range_key = self._get_dte_range_category(current_dte)
        
        if not dte_range_key:
            return {'error': f'DTE {current_dte} outside supported range (0-365)'}
        
        range_config = self.dte_range_configs[dte_range_key]
        historical_data = range_config['historical_iv_data']
        
        # Calculate current IV skew metrics (same as specific DTE)
        current_skew_analysis = self.calculate_comprehensive_iv_skew(
            option_chain_data, underlying_price, current_dte, symbol_type
        )
        
        # Store current data point for this DTE range
        current_data_point = {
            'timestamp': datetime.now(),
            'dte': current_dte,
            'dte_range': dte_range_key,
            'put_skew': current_skew_analysis['basic_skew_metrics']['put_skew'],
            'call_skew': current_skew_analysis['basic_skew_metrics']['call_skew'],
            'overall_skew': current_skew_analysis['basic_skew_metrics']['overall_skew'],
            'atm_iv': current_skew_analysis['basic_skew_metrics']['atm_iv'],
            'skew_ratio': current_skew_analysis['basic_skew_metrics']['skew_ratio'],
            'pattern_detected': current_skew_analysis.get('pattern_analysis', {}).get('primary_patterns', {}),
            'regime_implications': current_skew_analysis.get('regime_detection', {})
        }
        
        historical_data.append(current_data_point)
        
        if len(historical_data) >= 50:  # Minimum data for range percentile analysis
            # Calculate DTE range percentiles
            range_percentiles = self._calculate_dte_range_percentiles(
                historical_data, dte_range_key
            )
            
            # Learn parameters for this DTE range
            learned_params = self._learn_dte_range_parameters(
                historical_data, dte_range_key
            )
            
            # Classify regime based on DTE range patterns
            regime_classification = self._classify_dte_range_regime(
                current_data_point, range_percentiles, learned_params
            )
            
            range_config['skew_percentiles'] = range_percentiles
            range_config['learned_parameters'] = learned_params
            
            return {
                'dte': current_dte,
                'dte_range': dte_range_key,
                'dte_range_label': range_config['label'],
                'analysis_type': 'dte_range_iv_skew',
                'percentiles': range_percentiles,
                'learned_parameters': learned_params,
                'current_metrics': current_data_point,
                'regime_classification': regime_classification,
                'confidence': self._calculate_dte_range_confidence(
                    current_data_point, range_percentiles, len(historical_data)
                ),
                'weight_factor': range_config['weight_factor'],
                'data_quality': {
                    'historical_points': len(historical_data),
                    'data_sufficiency': 'sufficient' if len(historical_data) >= 100 else 'minimum'
                }
            }
        
        return {
            'dte': current_dte,
            'dte_range': dte_range_key,
            'dte_range_label': range_config['label'],
            'analysis_type': 'dte_range_iv_skew',
            'status': 'insufficient_data',
            'data_points': len(historical_data),
            'required_minimum': 50
        }

    def _get_dte_range_category(self, dte: int) -> str:
        """Determine which DTE range category the current DTE belongs to"""
        if 0 <= dte <= 7:
            return 'dte_0_to_7'
        elif 8 <= dte <= 30:
            return 'dte_8_to_30'
        elif 31 <= dte <= 365:
            return 'dte_31_plus'
        else:
            return None

    def _calculate_specific_dte_percentiles(self, historical_data, dte):
        """Calculate percentiles for specific DTE IV skew analysis"""
        put_skews = [d['put_skew'] for d in historical_data]
        call_skews = [d['call_skew'] for d in historical_data]
        overall_skews = [d['overall_skew'] for d in historical_data]
        atm_ivs = [d['atm_iv'] for d in historical_data]
        
        percentiles = {}
        
        # Put Skew Percentiles for this specific DTE
        percentiles['put_skew_percentiles'] = {
            f'dte_{dte}_put_skew_p10': float(np.percentile(put_skews, 10)),
            f'dte_{dte}_put_skew_p25': float(np.percentile(put_skews, 25)),
            f'dte_{dte}_put_skew_p50': float(np.percentile(put_skews, 50)),
            f'dte_{dte}_put_skew_p75': float(np.percentile(put_skews, 75)),
            f'dte_{dte}_put_skew_p90': float(np.percentile(put_skews, 90))
        }
        
        # Call Skew Percentiles for this specific DTE
        percentiles['call_skew_percentiles'] = {
            f'dte_{dte}_call_skew_p10': float(np.percentile(call_skews, 10)),
            f'dte_{dte}_call_skew_p25': float(np.percentile(call_skews, 25)),
            f'dte_{dte}_call_skew_p50': float(np.percentile(call_skews, 50)),
            f'dte_{dte}_call_skew_p75': float(np.percentile(call_skews, 75)),
            f'dte_{dte}_call_skew_p90': float(np.percentile(call_skews, 90))
        }
        
        # Overall Skew Percentiles for this specific DTE
        percentiles['overall_skew_percentiles'] = {
            f'dte_{dte}_overall_skew_p10': float(np.percentile(overall_skews, 10)),
            f'dte_{dte}_overall_skew_p25': float(np.percentile(overall_skews, 25)),
            f'dte_{dte}_overall_skew_p50': float(np.percentile(overall_skews, 50)),
            f'dte_{dte}_overall_skew_p75': float(np.percentile(overall_skews, 75)),
            f'dte_{dte}_overall_skew_p90': float(np.percentile(overall_skews, 90))
        }
        
        # ATM IV Percentiles for this specific DTE
        percentiles['atm_iv_percentiles'] = {
            f'dte_{dte}_atm_iv_p10': float(np.percentile(atm_ivs, 10)),
            f'dte_{dte}_atm_iv_p25': float(np.percentile(atm_ivs, 25)),
            f'dte_{dte}_atm_iv_p50': float(np.percentile(atm_ivs, 50)),
            f'dte_{dte}_atm_iv_p75': float(np.percentile(atm_ivs, 75)),
            f'dte_{dte}_atm_iv_p90': float(np.percentile(atm_ivs, 90))
        }
        
        return percentiles

    def _calculate_dte_range_percentiles(self, historical_data, dte_range_key):
        """Calculate percentiles for DTE range IV skew analysis"""
        put_skews = [d['put_skew'] for d in historical_data]
        call_skews = [d['call_skew'] for d in historical_data]
        overall_skews = [d['overall_skew'] for d in historical_data]
        atm_ivs = [d['atm_iv'] for d in historical_data]
        
        percentiles = {}
        
        # Put Skew Percentiles for this DTE range
        percentiles['put_skew_percentiles'] = {
            f'{dte_range_key}_put_skew_p10': float(np.percentile(put_skews, 10)),
            f'{dte_range_key}_put_skew_p25': float(np.percentile(put_skews, 25)),
            f'{dte_range_key}_put_skew_p50': float(np.percentile(put_skews, 50)),
            f'{dte_range_key}_put_skew_p75': float(np.percentile(put_skews, 75)),
            f'{dte_range_key}_put_skew_p90': float(np.percentile(put_skews, 90))
        }
        
        # Call Skew Percentiles for this DTE range
        percentiles['call_skew_percentiles'] = {
            f'{dte_range_key}_call_skew_p10': float(np.percentile(call_skews, 10)),
            f'{dte_range_key}_call_skew_p25': float(np.percentile(call_skews, 25)),
            f'{dte_range_key}_call_skew_p50': float(np.percentile(call_skews, 50)),
            f'{dte_range_key}_call_skew_p75': float(np.percentile(call_skews, 75)),
            f'{dte_range_key}_call_skew_p90': float(np.percentile(call_skews, 90))
        }
        
        # Overall Skew Percentiles for this DTE range
        percentiles['overall_skew_percentiles'] = {
            f'{dte_range_key}_overall_skew_p10': float(np.percentile(overall_skews, 10)),
            f'{dte_range_key}_overall_skew_p25': float(np.percentile(overall_skews, 25)),
            f'{dte_range_key}_overall_skew_p50': float(np.percentile(overall_skews, 50)),
            f'{dte_range_key}_overall_skew_p75': float(np.percentile(overall_skews, 75)),
            f'{dte_range_key}_overall_skew_p90': float(np.percentile(overall_skews, 90))
        }
        
        # ATM IV Percentiles for this DTE range
        percentiles['atm_iv_percentiles'] = {
            f'{dte_range_key}_atm_iv_p10': float(np.percentile(atm_ivs, 10)),
            f'{dte_range_key}_atm_iv_p25': float(np.percentile(atm_ivs, 25)),
            f'{dte_range_key}_atm_iv_p50': float(np.percentile(atm_ivs, 50)),
            f'{dte_range_key}_atm_iv_p75': float(np.percentile(atm_ivs, 75)),
            f'{dte_range_key}_atm_iv_p90': float(np.percentile(atm_ivs, 90))
        }
        
        return percentiles

    def cross_validate_dte_approaches(self, specific_results, range_results, current_dte):
        """Cross-validate specific DTE vs DTE range analysis approaches"""
        cross_validation = {}
        
        # Check if both analyses are available
        if (specific_results.get('status') == 'insufficient_data' or 
            range_results.get('status') == 'insufficient_data'):
            return {
                'validation_status': 'insufficient_data',
                'available_analyses': {
                    'specific_dte': specific_results.get('status') != 'insufficient_data',
                    'dte_range': range_results.get('status') != 'insufficient_data'
                }
            }
        
        # Compare regime classifications
        specific_regime = specific_results.get('regime_classification', {})
        range_regime = range_results.get('regime_classification', {})
        
        regime_agreement = self._calculate_regime_agreement(specific_regime, range_regime)
        
        # Compare percentile classifications
        specific_percentiles = specific_results.get('percentiles', {})
        range_percentiles = range_results.get('percentiles', {})
        
        percentile_consistency = self._calculate_percentile_consistency(
            specific_percentiles, range_percentiles, current_dte
        )
        
        # Compare confidence levels
        specific_confidence = specific_results.get('confidence', 0.0)
        range_confidence = range_results.get('confidence', 0.0)
        
        confidence_differential = abs(specific_confidence - range_confidence)
        
        cross_validation = {
            'validation_status': 'complete',
            'regime_agreement': regime_agreement,
            'percentile_consistency': percentile_consistency,
            'confidence_comparison': {
                'specific_dte_confidence': specific_confidence,
                'dte_range_confidence': range_confidence,
                'confidence_differential': confidence_differential,
                'agreement_level': 'high' if confidence_differential < 0.1 else 'medium' if confidence_differential < 0.2 else 'low'
            },
            'overall_consistency_score': self._calculate_overall_consistency(
                regime_agreement, percentile_consistency, confidence_differential
            )
        }
        
        return cross_validation

    def blend_dte_analysis_approaches(self, specific_results, range_results, cross_validation):
        """Blend specific DTE and range-based analysis for optimal results"""
        if cross_validation.get('validation_status') != 'complete':
            # Return the available analysis
            if specific_results.get('status') != 'insufficient_data':
                return {
                    'blended_approach': 'specific_dte_only',
                    'analysis_results': specific_results,
                    'confidence_adjustment': 0.8  # Reduced confidence for single approach
                }
            elif range_results.get('status') != 'insufficient_data':
                return {
                    'blended_approach': 'dte_range_only',
                    'analysis_results': range_results,
                    'confidence_adjustment': 0.8
                }
            else:
                return {'blended_approach': 'insufficient_data'}
        
        # Calculate blending weights based on consistency and confidence
        consistency_score = cross_validation['overall_consistency_score']
        
        # Higher consistency = more equal weighting
        # Lower consistency = favor the more confident analysis
        if consistency_score > 0.8:
            # High consistency - equal weighting
            specific_weight = 0.5
            range_weight = 0.5
        else:
            # Lower consistency - weight by confidence
            specific_confidence = cross_validation['confidence_comparison']['specific_dte_confidence']
            range_confidence = cross_validation['confidence_comparison']['dte_range_confidence']
            
            total_confidence = specific_confidence + range_confidence
            specific_weight = specific_confidence / total_confidence if total_confidence > 0 else 0.5
            range_weight = range_confidence / total_confidence if total_confidence > 0 else 0.5
        
        # Blend the analyses
        blended_analysis = {
            'blended_approach': 'weighted_combination',
            'weighting': {
                'specific_dte_weight': specific_weight,
                'dte_range_weight': range_weight,
                'consistency_score': consistency_score
            },
            'blended_regime_classification': self._blend_regime_classifications(
                specific_results.get('regime_classification', {}),
                range_results.get('regime_classification', {}),
                specific_weight, range_weight
            ),
            'blended_confidence': (
                specific_results.get('confidence', 0) * specific_weight +
                range_results.get('confidence', 0) * range_weight
            ),
            'recommendation': self._generate_blended_recommendation(
                specific_results, range_results, specific_weight, range_weight, consistency_score
            )
        }
        
        return blended_analysis

    def recommend_analysis_approach(self, cross_validation):
        """Recommend the optimal analysis approach based on validation results"""
        if cross_validation.get('validation_status') != 'complete':
            available_analyses = cross_validation.get('available_analyses', {})
            if available_analyses.get('specific_dte'):
                return {
                    'recommended_approach': 'specific_dte',
                    'reason': 'Specific DTE analysis available, range analysis insufficient data',
                    'confidence': 0.7
                }
            elif available_analyses.get('dte_range'):
                return {
                    'recommended_approach': 'dte_range',
                    'reason': 'DTE range analysis available, specific DTE insufficient data',
                    'confidence': 0.7
                }
            else:
                return {
                    'recommended_approach': 'none',
                    'reason': 'Both approaches have insufficient data',
                    'confidence': 0.0
                }
        
        consistency_score = cross_validation.get('overall_consistency_score', 0.0)
        confidence_comparison = cross_validation.get('confidence_comparison', {})
        
        if consistency_score > 0.8:
            return {
                'recommended_approach': 'blended',
                'reason': 'High consistency between approaches, blended analysis recommended',
                'confidence': 0.95,
                'blending_ratio': 'equal_weight'
            }
        elif consistency_score > 0.6:
            return {
                'recommended_approach': 'blended',
                'reason': 'Moderate consistency between approaches, confidence-weighted blending recommended',
                'confidence': 0.85,
                'blending_ratio': 'confidence_weighted'
            }
        else:
            # Low consistency - recommend the higher confidence approach
            if confidence_comparison.get('specific_dte_confidence', 0) > confidence_comparison.get('dte_range_confidence', 0):
                return {
                    'recommended_approach': 'specific_dte',
                    'reason': 'Low consistency, specific DTE analysis has higher confidence',
                    'confidence': confidence_comparison.get('specific_dte_confidence', 0) * 0.9
                }
            else:
                return {
                    'recommended_approach': 'dte_range',
                    'reason': 'Low consistency, DTE range analysis has higher confidence',
                    'confidence': confidence_comparison.get('dte_range_confidence', 0) * 0.9
                }
```

### **Enhanced IV Skew Analysis with Dual DTE Integration**
```python
class EnhancedIVSkewAnalysisEngine(IVSkewAnalysisEngine, DualDTEIVSkewAnalysisEngine):
    def __init__(self):
        super().__init__()
        DualDTEIVSkewAnalysisEngine.__init__(self)
        
        # Enhanced integration configuration
        self.dual_dte_integration_config = {
            'specific_dte_priority': True,      # Prioritize specific DTE when available
            'range_fallback_enabled': True,    # Use range analysis as fallback
            'cross_validation_threshold': 0.7, # Minimum consistency for blending
            'confidence_boost_factor': 1.1,    # Boost confidence when both approaches agree
            'percentile_analysis_enhanced': True
        }
        
        # Performance tracking for dual DTE approach
        self.dual_dte_performance = {
            'specific_dte_accuracy': {},       # Track accuracy per specific DTE
            'range_accuracy': {},              # Track accuracy per DTE range
            'blended_accuracy': 0.0,           # Overall blended approach accuracy
            'recommendation_success_rate': 0.0 # Success rate of approach recommendations
        }
    
    def perform_comprehensive_dual_dte_iv_analysis(self, option_chain_data, underlying_price, 
                                                  current_dte, symbol_type, market_context=None):
        """
        Perform comprehensive IV skew analysis using dual DTE approach
        
        This method integrates:
        1. Traditional IV skew analysis (existing)
        2. Specific DTE percentile analysis (new)
        3. DTE range percentile analysis (new)
        4. Cross-validation and blending (new)
        5. Enhanced regime classification (enhanced)
        """
        analysis_start = time.time()
        
        # Step 1: Traditional IV Skew Analysis (existing functionality)
        traditional_iv_analysis = self.calculate_comprehensive_iv_skew(
            option_chain_data, underlying_price, current_dte, symbol_type
        )
        
        # Step 2: Dual DTE Analysis (new functionality)
        dual_dte_analysis = self.analyze_dual_dte_iv_skew(
            option_chain_data, underlying_price, current_dte, symbol_type
        )
        
        # Step 3: Enhanced Pattern Recognition with DTE Context
        enhanced_pattern_analysis = self.analyze_dte_specific_patterns(
            traditional_iv_analysis, dual_dte_analysis, current_dte
        )
        
        # Step 4: Enhanced Regime Detection with Dual DTE
        enhanced_regime_detection = self.detect_enhanced_iv_regime_with_dte(
            traditional_iv_analysis, dual_dte_analysis, enhanced_pattern_analysis, 
            current_dte, symbol_type
        )
        
        # Step 5: Confidence scoring with dual approach validation
        enhanced_confidence = self.calculate_enhanced_confidence_with_dte(
            traditional_iv_analysis, dual_dte_analysis, enhanced_regime_detection
        )
        
        # Step 6: Generate comprehensive recommendations
        comprehensive_recommendations = self.generate_comprehensive_iv_recommendations(
            dual_dte_analysis, enhanced_regime_detection, enhanced_confidence
        )
        
        analysis_time = time.time() - analysis_start
        
        return {
            'timestamp': datetime.now().isoformat(),
            'component': 'Component 4: Enhanced IV Skew Analysis with Dual DTE',
            'dte': current_dte,
            'analysis_type': 'comprehensive_dual_dte_iv_skew',
            
            # Traditional analysis (maintained for compatibility)
            'traditional_iv_analysis': traditional_iv_analysis,
            
            # New dual DTE analysis
            'dual_dte_analysis': dual_dte_analysis,
            
            # Enhanced analyses
            'enhanced_pattern_analysis': enhanced_pattern_analysis,
            'enhanced_regime_detection': enhanced_regime_detection,
            'enhanced_confidence': enhanced_confidence,
            'comprehensive_recommendations': comprehensive_recommendations,
            
            # Performance metrics
            'analysis_time_ms': analysis_time * 1000,
            'performance_target_met': analysis_time < 0.2,  # <200ms target
            
            # Component health
            'component_health': {
                'traditional_analysis_active': 'error' not in traditional_iv_analysis,
                'dual_dte_analysis_active': dual_dte_analysis.get('specific_dte_analysis', {}).get('status') != 'insufficient_data',
                'pattern_recognition_active': len(enhanced_pattern_analysis.get('detected_patterns', {})) > 0,
                'regime_detection_active': enhanced_regime_detection.get('final_assessment', {}).get('primary_regime') is not None
            }
        }
    
    def analyze_dte_specific_patterns(self, traditional_analysis, dual_dte_analysis, current_dte):
        """Analyze IV patterns with DTE-specific context"""
        dte_specific_patterns = {}
        
        # Get DTE-specific percentiles
        specific_dte_results = dual_dte_analysis.get('specific_dte_analysis', {})
        range_results = dual_dte_analysis.get('dte_range_analysis', {})
        
        # Enhanced pattern detection with DTE context
        if specific_dte_results.get('status') != 'insufficient_data':
            specific_patterns = self._detect_dte_specific_patterns(
                traditional_analysis, specific_dte_results, current_dte
            )
            dte_specific_patterns['specific_dte_patterns'] = specific_patterns
        
        # Enhanced pattern detection with range context
        if range_results.get('status') != 'insufficient_data':
            range_patterns = self._detect_dte_range_patterns(
                traditional_analysis, range_results, current_dte
            )
            dte_specific_patterns['dte_range_patterns'] = range_patterns
        
        # Cross-pattern validation
        if 'specific_dte_patterns' in dte_specific_patterns and 'dte_range_patterns' in dte_specific_patterns:
            pattern_consistency = self._validate_pattern_consistency(
                dte_specific_patterns['specific_dte_patterns'],
                dte_specific_patterns['dte_range_patterns']
            )
            dte_specific_patterns['pattern_consistency'] = pattern_consistency
        
        return dte_specific_patterns
    
    def detect_enhanced_iv_regime_with_dte(self, traditional_analysis, dual_dte_analysis, 
                                          pattern_analysis, current_dte, symbol_type):
        """Enhanced IV regime detection incorporating dual DTE analysis"""
        enhanced_regime_detection = {}
        
        # Traditional regime scores (existing)
        traditional_regime_scores = traditional_analysis.get('regime_detection', {}).get('regime_scores', {})
        
        # DTE-enhanced regime scoring
        dte_enhanced_scores = self._calculate_dte_enhanced_regime_scores(
            traditional_regime_scores, dual_dte_analysis, pattern_analysis, current_dte
        )
        
        # Regime validation with dual DTE context
        regime_validation = self._validate_regimes_with_dte_context(
            dte_enhanced_scores, dual_dte_analysis, current_dte
        )
        
        # Final regime assessment with DTE weighting
        final_assessment = self._generate_dte_weighted_regime_assessment(
            dte_enhanced_scores, regime_validation, current_dte, symbol_type
        )
        
        enhanced_regime_detection = {
            'traditional_regime_scores': traditional_regime_scores,
            'dte_enhanced_scores': dte_enhanced_scores,
            'regime_validation': regime_validation,
            'final_assessment': final_assessment,
            'confidence_level': self._calculate_dte_enhanced_confidence(
                final_assessment, regime_validation, dual_dte_analysis
            )
        }
        
        return enhanced_regime_detection
    
    def generate_comprehensive_iv_recommendations(self, dual_dte_analysis, regime_detection, confidence):
        """Generate comprehensive recommendations based on dual DTE IV analysis"""
        recommendations = {}
        
        # Approach recommendations
        recommended_approach = dual_dte_analysis.get('recommended_approach', {})
        recommendations['analysis_approach'] = recommended_approach
        
        # DTE-specific recommendations
        if dual_dte_analysis.get('specific_dte_analysis', {}).get('status') != 'insufficient_data':
            dte_recommendations = self._generate_specific_dte_recommendations(
                dual_dte_analysis['specific_dte_analysis'], regime_detection
            )
            recommendations['dte_specific'] = dte_recommendations
        
        # Range-based recommendations
        if dual_dte_analysis.get('dte_range_analysis', {}).get('status') != 'insufficient_data':
            range_recommendations = self._generate_dte_range_recommendations(
                dual_dte_analysis['dte_range_analysis'], regime_detection
            )
            recommendations['dte_range'] = range_recommendations
        
        # Trading implications based on IV skew and DTE
        trading_implications = self._generate_iv_trading_implications(
            dual_dte_analysis, regime_detection, confidence
        )
        recommendations['trading_implications'] = trading_implications
        
        # Risk management based on IV patterns and DTE
        risk_management = self._generate_iv_risk_management(
            dual_dte_analysis, regime_detection, confidence
        )
        recommendations['risk_management'] = risk_management
        
        return recommendations
```

### **IV Parameter Learning Engine**
```python
class IVParameterLearningEngine:
            'optimization_objectives': ['accuracy', 'sharpe_ratio', 'regime_detection_accuracy'],
            'parameter_ranges': {
                'skew_thresholds': {'min': -0.25, 'max': 0.10, 'step': 0.01},
                'atm_iv_thresholds': {'min': 0.10, 'max': 0.80, 'step': 0.02},
                'regime_confidence': {'min': 0.3, 'max': 0.95, 'step': 0.05}
            },
            'lookback_periods': [30, 60, 90, 180, 365]  # Days of historical data
        }
        
        # Performance tracking
        self.performance_metrics = {
            'regime_detection_accuracy': {},
            'parameter_stability': {},
            'prediction_confidence': {},
            'false_signal_rate': {}
        }
    
    def learn_adaptive_iv_parameters(self, historical_iv_data, historical_market_outcomes, 
                                   current_dte, symbol_type, learning_mode='both'):
        """Learn optimal IV analysis parameters from historical performance"""
        learning_results = {}
        
        # 1. Prepare historical data for learning
        prepared_data = self.prepare_historical_iv_data(
            historical_iv_data, historical_market_outcomes, current_dte, symbol_type
        )
        
        # 2. DTE-specific learning if requested
        if learning_mode in ['dte_specific', 'both']:
            dte_specific_params = self.learn_dte_specific_iv_parameters(
                prepared_data, current_dte, symbol_type
            )
            learning_results['dte_specific'] = dte_specific_params
        
        # 3. All-days learning if requested
        if learning_mode in ['all_days', 'both']:
            all_days_params = self.learn_all_days_iv_parameters(
                prepared_data, symbol_type
            )
            learning_results['all_days'] = all_days_params
        
        # 4. Blend learning approaches if both selected
        if learning_mode == 'both':
            blended_params = self.blend_iv_learning_approaches(
                learning_results['dte_specific'], learning_results['all_days']
            )
            learning_results['blended'] = blended_params
        
        # 5. Validate learned parameters
        validation_results = self.validate_learned_iv_parameters(
            learning_results, prepared_data
        )
        
        # 6. Generate final optimized parameters
        final_params = self.generate_final_iv_parameters(
            learning_results, validation_results, learning_mode
        )
        
        return {
            'learning_results': learning_results,
            'validation_results': validation_results,
            'final_optimized_parameters': final_params,
            'learning_confidence': self.calculate_learning_confidence(validation_results)
        }
    
    def learn_dte_specific_iv_parameters(self, prepared_data, current_dte, symbol_type):
        """Learn IV parameters specific to current DTE"""
        dte_filtered_data = prepared_data[prepared_data['dte'] == current_dte]
        
        if len(dte_filtered_data) < 20:  # Insufficient data
            return self.get_default_iv_parameters(symbol_type)
        
        # Optimize parameters for this specific DTE
        optimized_params = {}
        
        # Optimize skew thresholds
        optimized_params['skew_thresholds'] = self.optimize_skew_thresholds(
            dte_filtered_data, symbol_type
        )
        
        # Optimize regime detection thresholds
        optimized_params['regime_thresholds'] = self.optimize_regime_thresholds(
            dte_filtered_data, symbol_type
        )
        
        # Optimize confidence scoring parameters
        optimized_params['confidence_params'] = self.optimize_confidence_parameters(
            dte_filtered_data, symbol_type
        )
        
        # Calculate performance metrics
        optimized_params['performance_metrics'] = self.calculate_dte_performance_metrics(
            dte_filtered_data, optimized_params
        )
        
        return optimized_params
    
    def optimize_skew_thresholds(self, data, symbol_type):
        """Optimize IV skew classification thresholds"""
        from scipy.optimize import differential_evolution
        
        def objective_function(params):
            put_skew_threshold, call_skew_threshold, overall_skew_threshold = params
            
            # Apply thresholds and calculate accuracy
            accuracy = self.calculate_threshold_accuracy(
                data, put_skew_threshold, call_skew_threshold, overall_skew_threshold
            )
            
            return -accuracy  # Minimize negative accuracy
        
        # Parameter bounds
        bounds = [
            (-0.25, -0.01),  # Put skew threshold range
            (0.01, 0.15),    # Call skew threshold range
            (-0.20, 0.20)    # Overall skew threshold range
        ]
        
        # Optimize
        result = differential_evolution(objective_function, bounds, maxiter=100, seed=42)
        
        return {
            'put_skew_threshold': result.x[0],
            'call_skew_threshold': result.x[1], 
            'overall_skew_threshold': result.x[2],
            'optimization_score': -result.fun
        }
    
    def calculate_threshold_accuracy(self, data, put_threshold, call_threshold, overall_threshold):
        """Calculate accuracy of threshold-based classification"""
        correct_classifications = 0
        total_classifications = len(data)
        
        for _, row in data.iterrows():
            # Apply thresholds to classify regime
            predicted_regime = self.classify_regime_with_thresholds(
                row, put_threshold, call_threshold, overall_threshold
            )
            
            # Check if prediction matches actual outcome
            if predicted_regime == row['actual_regime']:
                correct_classifications += 1
        
        return correct_classifications / total_classifications if total_classifications > 0 else 0.0
    
    def get_default_iv_parameters(self, symbol_type):
        """Get default IV parameters when insufficient learning data"""
        default_params = {
            'NIFTY': {
                'skew_thresholds': {'put': -0.08, 'call': 0.03, 'overall': -0.05},
                'regime_thresholds': {'confidence_min': 0.6, 'validation_threshold': 0.7},
                'confidence_params': {'base_confidence': 0.7, 'adjustment_factor': 1.0}
            },
            'BANKNIFTY': {
                'skew_thresholds': {'put': -0.12, 'call': 0.04, 'overall': -0.08},
                'regime_thresholds': {'confidence_min': 0.65, 'validation_threshold': 0.75},
                'confidence_params': {'base_confidence': 0.75, 'adjustment_factor': 1.1}
            },
            'STOCKS': {
                'skew_thresholds': {'put': -0.18, 'call': 0.06, 'overall': -0.12},
                'regime_thresholds': {'confidence_min': 0.55, 'validation_threshold': 0.65},
                'confidence_params': {'base_confidence': 0.65, 'adjustment_factor': 0.9}
            }
        }
        
        return default_params.get(symbol_type, default_params['NIFTY'])
```

---

## **5. Integration with Market Regime Framework**

### **IV Skew Regime Contribution Engine**
```python
class IVSkewRegimeContributionEngine:
    def __init__(self):
        # Contribution weights for each regime (learned from historical data)
        self.regime_contribution_weights = {
            'LVLD': 0.20,  # IV skew provides good confirmation for low vol environments
            'HVC': 0.25,   # IV skew excellent predictor of volatility convergence
            'VCPE': 0.30,  # IV smile patterns key indicator for this regime
            'TBVE': 0.15,  # IV skew secondary indicator for trend breaks
            'TBVS': 0.20,  # Reverse skew good indicator for squeeze
            'SCGS': 0.35,  # IV patterns excellent for gamma squeeze detection
            'PSED': 0.25,  # Term structure inversion key for this regime
            'CBV': 0.10    # IV skew less reliable for correlation breaks
        }
        
        # Integration parameters with other components
        self.integration_params = {
            'oi_analysis_weight': 0.40,     # Weight vs OI analysis (Component 3)
            'greeks_weight': 0.35,          # Weight vs Greeks analysis (Component 2)  
            'iv_skew_weight': 0.25,         # Weight of IV skew analysis
            'minimum_confirmation': 0.6,    # Minimum other component confirmation needed
            'override_threshold': 0.85      # Threshold for IV skew to override other signals
        }
    
    def calculate_regime_contributions(self, iv_skew_analysis, oi_analysis_results, 
                                     greeks_analysis_results, market_data):
        """Calculate IV skew contributions to overall regime classification"""
        contributions = {}
        
        # Extract IV regime detection results
        iv_regime_scores = iv_skew_analysis['regime_detection']['regime_scores']
        iv_confidence = iv_skew_analysis['regime_detection']['confidence_level']
        
        # Calculate base contributions for each regime
        for regime, score in iv_regime_scores.items():
            # Base contribution from IV analysis
            base_contribution = score * self.regime_contribution_weights.get(regime, 0.1)
            
            # Adjust based on confidence level
            confidence_adjusted = base_contribution * iv_confidence
            
            # Cross-validation with other components
            oi_confirmation = self.get_oi_confirmation(regime, oi_analysis_results)
            greeks_confirmation = self.get_greeks_confirmation(regime, greeks_analysis_results)
            
            # Calculate integrated contribution
            integrated_contribution = self.calculate_integrated_contribution(
                confidence_adjusted, oi_confirmation, greeks_confirmation, regime
            )
            
            contributions[regime] = {
                'base_iv_contribution': base_contribution,
                'confidence_adjusted': confidence_adjusted,
                'oi_confirmation': oi_confirmation,
                'greeks_confirmation': greeks_confirmation,
                'final_contribution': integrated_contribution,
                'contribution_confidence': self.calculate_contribution_confidence(
                    iv_confidence, oi_confirmation, greeks_confirmation
                )
            }
        
        return contributions
    
    def calculate_integrated_contribution(self, iv_contribution, oi_confirmation, 
                                        greeks_confirmation, regime):
        """Calculate final integrated contribution with cross-validation"""
        
        # If IV signal is very strong, allow it more influence
        if iv_contribution > self.integration_params['override_threshold']:
            # Strong IV signal gets higher weight
            integrated = (
                iv_contribution * 0.6 +
                oi_confirmation * 0.25 +
                greeks_confirmation * 0.15
            )
        else:
            # Normal weighting
            integrated = (
                iv_contribution * self.integration_params['iv_skew_weight'] +
                oi_confirmation * self.integration_params['oi_analysis_weight'] +
                greeks_confirmation * self.integration_params['greeks_weight']
            )
        
        # Apply minimum confirmation requirement
        min_other_confirmation = max(oi_confirmation, greeks_confirmation)
        if min_other_confirmation < self.integration_params['minimum_confirmation']:
            # Reduce contribution if other components don't confirm
            integrated *= 0.7
        
        return min(integrated, 1.0)
    
    def get_oi_confirmation(self, regime, oi_analysis_results):
        """Get confirmation level from OI analysis for specific regime"""
        if not oi_analysis_results or 'integrated_signals' not in oi_analysis_results:
            return 0.5  # Neutral if no OI data
        
        # Extract relevant OI signals for regime confirmation
        oi_signals = oi_analysis_results['integrated_signals']
        regime_contribution = oi_analysis_results.get('overall_market_regime_contribution', {})
        
        # Get regime-specific confirmation
        confirmation = regime_contribution.get(regime, 0.0)
        
        # Boost confirmation if OI signals align with IV signals
        if regime in ['HVC', 'VCPE', 'SCGS']:  # Volatility-related regimes
            if oi_signals.get('overall_direction') == 'bullish' and regime in ['HVC', 'VCPE']:
                confirmation += 0.1
            elif oi_signals.get('overall_direction') == 'bearish' and regime == 'SCGS':
                confirmation += 0.1
        
        return min(confirmation, 1.0)
    
    def get_greeks_confirmation(self, regime, greeks_analysis_results):
        """Get confirmation level from Greeks analysis for specific regime"""
        if not greeks_analysis_results:
            return 0.5  # Neutral if no Greeks data
        
        # Extract Greeks signals relevant to regime
        # This would integrate with Component 2 results
        greeks_regime_contribution = greeks_analysis_results.get('regime_contribution', {})
        
        confirmation = greeks_regime_contribution.get(regime, 0.0)
        
        # Specific regime-Greeks relationships
        if regime == 'SCGS':  # Gamma squeeze regime
            gamma_signals = greeks_analysis_results.get('gamma_analysis', {})
            if gamma_signals.get('gamma_squeeze_detected', False):
                confirmation += 0.15
        
        elif regime in ['TBVE', 'TBVS']:  # Trend break regimes
            delta_signals = greeks_analysis_results.get('delta_analysis', {})
            if delta_signals.get('delta_momentum', 0) > 0.1:
                confirmation += 0.1
        
        return min(confirmation, 1.0)
```

---

## **Summary**

Component 4: Enhanced IV Skew Analysis System with Dual DTE Framework provides:

### **Core Features:**
1. **Comprehensive IV Skew Calculation** across ATM ±7 strikes with adaptive strike selection
2. **Dual DTE Analysis Framework** supporting both specific DTE (dte=0, dte=1, dte=7, etc.) AND DTE ranges (dte_0_to_7, dte_8_to_30, dte_31_plus)
3. **Advanced Pattern Recognition** for normal skew, steep put skew, flat skew, reverse skew, and volatility smile patterns with DTE-specific context
4. **Regime-Specific IV Analysis** with dedicated scoring algorithms for all 8 market regimes enhanced by DTE analysis
5. **Cross-Validation Framework** between specific DTE and DTE range approaches with intelligent blending
6. **Adaptive Parameter Learning** supporting both DTE-specific and all-days learning modes with enhanced accuracy tracking
7. **Historical Performance Optimization** with multi-objective parameter tuning across different DTE categories
8. **Symbol-Specific Calibration** for NIFTY, BANKNIFTY, and individual stocks with DTE-aware adjustments
9. **Cross-Component Integration** with OI analysis (Component 3) and Greeks analysis (Component 2) enhanced by DTE context
10. **Real-Time Pattern Classification** with confidence scoring and validation across multiple DTE perspectives
11. **Term Structure Analysis** for front-month vs back-month IV relationships with DTE-specific percentiles
12. **Market Implication Assessment** with directional bias and sentiment analysis enhanced by DTE-specific patterns

### **New Dual DTE Capabilities:**
13. **Specific DTE Percentile Analysis** - When user requests "dte=0 percentile", analyzes percentiles specifically for dte=0 data with 252-day rolling history
14. **DTE Range Percentile Analysis** - Supports weekly (dte_0_to_7), monthly (dte_8_to_30), and far month (dte_31_plus) analysis cycles
15. **Intelligent Approach Recommendation** - Automatically recommends optimal analysis approach based on data availability and consistency
16. **Enhanced Confidence Scoring** - Dual approach validation boosts confidence when both specific DTE and range analysis agree
17. **Production-Grade Integration** - Maintains full backward compatibility while adding advanced DTE-specific insights

### **Key Innovation:**
The system combines **real-time IV skew pattern recognition** with **adaptive historical learning** and **production-grade IV surface analysis** (based on existing implementation) to identify market regime transitions before they become apparent in price action, providing significant predictive capability for institutional-level trading strategies.

### **Production Integration Note:**
This component enhances the existing comprehensive IV analysis suite including `iv_percentile_analyzer.py`, `iv_skew_analyzer.py`, and `iv_surface_analyzer.py` with advanced adaptive learning and regime-specific optimizations.

### **Integration Benefits:**
- **25% contribution weight** to overall market regime classification
- **Cross-validation** with OI and Greeks analysis for enhanced accuracy
- **Override capability** for strong IV signals (>85% confidence threshold)
- **Dynamic weighting** based on market conditions and component confirmations

The IV Skew Analysis System represents a sophisticated approach to volatility-based regime detection that complements the OI and Greeks analysis components for comprehensive market regime classification.

---

## **6. Production-Grade Enhancements Based on Existing Implementation**

### **IV Percentile Analysis Integration**
```python
class ProductionIVPercentileAnalyzer:
    def __init__(self):
        # Based on existing iv_percentile_analyzer.py implementation
        self.dte_categories = {
            'dte_0_to_7': (0, 7),      # DTE 0-7 (Weekly expiry cycle)
            'dte_8_to_30': (8, 30),   # DTE 8-30 (Monthly expiry cycle)  
            'dte_31_plus': (31, 365)   # DTE 31+ (Far month expiries)
        }
        
        # IV percentile thresholds (calibrated for Indian market)
        self.iv_percentile_thresholds = {
            'extremely_low': 10,      # Bottom 10th percentile
            'very_low': 25,           # 10th-25th percentile
            'low': 40,                # 25th-40th percentile
            'neutral': 60,            # 40th-60th percentile
            'high': 75,               # 60th-75th percentile
            'very_high': 90,          # 75th-90th percentile
            'extremely_high': 100     # Top 10th percentile
        }
        
        # Historical lookback for percentile calculation
        self.historical_lookback = 252  # 1 trading year
        
    def calculate_iv_percentile_regime(self, current_iv_data, historical_iv_data, current_dte):
        """Calculate IV percentile-based regime classification"""
        # Determine DTE category
        dte_category = self.classify_dte_category(current_dte)
        
        # Filter historical data for same DTE category
        relevant_historical = self.filter_historical_by_dte(
            historical_iv_data, dte_category, self.historical_lookback
        )
        
        # Calculate percentile rankings
        iv_percentiles = {}
        
        for strike_type in ['atm', 'otm_calls', 'otm_puts']:
            current_iv = current_iv_data.get(strike_type, 0.2)  # Default 20% if missing
            
            # Handle extreme values (as in production implementation)
            normalized_current_iv = self.normalize_extreme_iv_values(current_iv)
            
            # Calculate percentile rank
            if len(relevant_historical) > 50:  # Sufficient data
                percentile_rank = self.calculate_percentile_rank(
                    normalized_current_iv, relevant_historical[strike_type]
                )
            else:
                percentile_rank = 50  # Neutral if insufficient data
            
            # Classify IV regime
            iv_regime = self.classify_iv_regime_by_percentile(percentile_rank)
            
            iv_percentiles[strike_type] = {
                'current_iv': normalized_current_iv,
                'percentile_rank': percentile_rank,
                'iv_regime': iv_regime,
                'confidence': self.calculate_iv_confidence(relevant_historical, current_iv)
            }
        
        return iv_percentiles
    
    def normalize_extreme_iv_values(self, iv_value):
        """Normalize extreme IV values (from production implementation)"""
        # Handle extremely low IV (like 0.01 CE IV)
        if iv_value < 0.02:
            normalized_iv = max(iv_value, 0.02)  # Floor at 2%
        # Handle extremely high IV (like 60+ PE IV in stressed conditions)  
        elif iv_value > 2.0:
            normalized_iv = min(iv_value, 2.0)   # Ceiling at 200%
        else:
            normalized_iv = iv_value
            
        return normalized_iv
    
    def classify_iv_regime_by_percentile(self, percentile_rank):
        """Classify IV regime based on percentile ranking"""
        if percentile_rank <= self.iv_percentile_thresholds['extremely_low']:
            return 'extremely_low_vol'
        elif percentile_rank <= self.iv_percentile_thresholds['very_low']:
            return 'very_low_vol'
        elif percentile_rank <= self.iv_percentile_thresholds['low']:
            return 'low_vol'
        elif percentile_rank <= self.iv_percentile_thresholds['neutral']:
            return 'neutral_vol'
        elif percentile_rank <= self.iv_percentile_thresholds['high']:
            return 'high_vol'
        elif percentile_rank <= self.iv_percentile_thresholds['very_high']:
            return 'very_high_vol'
        else:
            return 'extremely_high_vol'
```

### **Enhanced IV Surface Analysis**
```python
class ProductionIVSurfaceAnalyzer:
    def __init__(self):
        # Based on existing iv_surface_analyzer.py implementation
        self.surface_analysis_config = {
            'strike_range_pct': 0.10,     # ±10% from ATM for surface analysis
            'min_strikes_required': 5,    # Minimum strikes for valid surface
            'surface_smoothing_factor': 0.1,
            'curvature_detection_threshold': 0.05,
            'analysis_performance_target': 200  # <200ms target
        }
        
        # Surface pattern classification (7 levels from production)
        self.surface_patterns = {
            'normal_skew': 'Standard put skew with call smile',
            'steep_skew': 'Elevated put skew indicating fear',
            'volatility_smile': 'Symmetric smile indicating uncertainty',
            'reverse_skew': 'Call skew higher than put skew',
            'flat_surface': 'Minimal skew across strikes',
            'convex_smile': 'Strong curvature in IV surface',
            'irregular_surface': 'Non-standard IV patterns'
        }
    
    def analyze_3d_iv_surface(self, option_chain_data, underlying_price):
        """Analyze 3D volatility surface across strikes and expiries"""
        surface_analysis = {}
        
        # 1. Construct IV surface matrix
        iv_surface_matrix = self.construct_iv_surface_matrix(
            option_chain_data, underlying_price
        )
        
        # 2. Calculate surface curvature
        surface_curvature = self.calculate_surface_curvature(iv_surface_matrix)
        
        # 3. Detect surface patterns
        detected_patterns = self.detect_surface_patterns(
            iv_surface_matrix, surface_curvature
        )
        
        # 4. Analyze surface stability
        surface_stability = self.analyze_surface_stability(iv_surface_matrix)
        
        # 5. Generate regime implications
        regime_implications = self.generate_surface_regime_implications(
            detected_patterns, surface_curvature, surface_stability
        )
        
        surface_analysis = {
            'iv_surface_matrix': iv_surface_matrix,
            'surface_curvature': surface_curvature,
            'detected_patterns': detected_patterns,
            'surface_stability': surface_stability,
            'regime_implications': regime_implications,
            'analysis_confidence': self.calculate_surface_analysis_confidence(
                iv_surface_matrix, detected_patterns
            )
        }
        
        return surface_analysis
    
    def construct_iv_surface_matrix(self, option_chain_data, underlying_price):
        """Construct 3D IV surface matrix"""
        # Extract strikes within ±10% range
        strike_range = underlying_price * self.surface_analysis_config['strike_range_pct']
        lower_bound = underlying_price - strike_range
        upper_bound = underlying_price + strike_range
        
        surface_matrix = {
            'call_surface': {},
            'put_surface': {},
            'strikes': [],
            'expiries': []
        }
        
        # Process call options
        for strike_str, option_data in option_chain_data.get('CE', {}).items():
            strike = float(strike_str)
            if lower_bound <= strike <= upper_bound:
                iv = option_data.get('IV', 0)
                if iv > 0:  # Valid IV data
                    surface_matrix['call_surface'][strike] = {
                        'iv': self.normalize_extreme_iv_values(iv),
                        'volume': option_data.get('volume', 0),
                        'oi': option_data.get('OI', 0),
                        'moneyness': (strike - underlying_price) / underlying_price
                    }
                    if strike not in surface_matrix['strikes']:
                        surface_matrix['strikes'].append(strike)
        
        # Process put options
        for strike_str, option_data in option_chain_data.get('PE', {}).items():
            strike = float(strike_str)
            if lower_bound <= strike <= upper_bound:
                iv = option_data.get('IV', 0)
                if iv > 0:  # Valid IV data
                    surface_matrix['put_surface'][strike] = {
                        'iv': self.normalize_extreme_iv_values(iv),
                        'volume': option_data.get('volume', 0),
                        'oi': option_data.get('OI', 0),
                        'moneyness': (strike - underlying_price) / underlying_price
                    }
                    if strike not in surface_matrix['strikes']:
                        surface_matrix['strikes'].append(strike)
        
        # Sort strikes for analysis
        surface_matrix['strikes'].sort()
        
        return surface_matrix
```

### **Dynamic VIX-Based Threshold Optimization**
```python
class DynamicIVThresholdOptimizer:
    def __init__(self):
        # Based on existing dynamic threshold optimization
        self.optimization_config = {
            'vix_based_scaling': True,
            'hysteresis_factor': 0.05,    # Prevents oscillation
            'optimization_target_ms': 30,  # <30ms optimization target
            'confidence_threshold': 0.75   # Minimum confidence for threshold change
        }
        
        # VIX-based scaling factors (from production implementation)
        self.vix_scaling_factors = {
            'low_vix': {'threshold': 15, 'iv_scaling': 0.8},      # VIX < 15
            'normal_vix': {'threshold': 25, 'iv_scaling': 1.0},   # VIX 15-25
            'high_vix': {'threshold': 35, 'iv_scaling': 1.3},     # VIX 25-35
            'extreme_vix': {'threshold': 50, 'iv_scaling': 1.8}   # VIX > 35
        }
    
    def optimize_iv_thresholds_dynamically(self, current_vix, historical_performance, 
                                          current_thresholds):
        """Dynamically optimize IV thresholds based on VIX and performance"""
        # Determine VIX regime
        vix_regime = self.classify_vix_regime(current_vix)
        
        # Get scaling factor for current VIX regime
        scaling_factor = self.vix_scaling_factors[vix_regime]['iv_scaling']
        
        # Calculate optimized thresholds
        optimized_thresholds = {}
        
        for threshold_type, current_value in current_thresholds.items():
            # Apply VIX-based scaling
            vix_adjusted = current_value * scaling_factor
            
            # Apply hysteresis to prevent oscillation
            hysteresis_adjusted = self.apply_hysteresis(
                vix_adjusted, current_value, self.optimization_config['hysteresis_factor']
            )
            
            # Validate with historical performance
            performance_validated = self.validate_threshold_with_performance(
                hysteresis_adjusted, historical_performance.get(threshold_type, {})
            )
            
            optimized_thresholds[threshold_type] = performance_validated
        
        return {
            'optimized_thresholds': optimized_thresholds,
            'vix_regime': vix_regime,
            'scaling_applied': scaling_factor,
            'optimization_confidence': self.calculate_optimization_confidence(
                historical_performance, optimized_thresholds
            )
        }
    
    def classify_vix_regime(self, current_vix):
        """Classify VIX regime for threshold scaling"""
        if current_vix < self.vix_scaling_factors['low_vix']['threshold']:
            return 'low_vix'
        elif current_vix < self.vix_scaling_factors['normal_vix']['threshold']:
            return 'normal_vix'  
        elif current_vix < self.vix_scaling_factors['high_vix']['threshold']:
            return 'high_vix'
        else:
            return 'extreme_vix'
```

### **Fear/Greed Analysis Integration**
```python
class ProductionFearGreedAnalyzer:
    def __init__(self):
        # Based on existing fear/greed analysis implementation
        self.fear_greed_thresholds = {
            'extreme_fear': -0.15,      # Steep put skew
            'fear': -0.08,              # Moderate put skew
            'neutral': 0.03,            # Balanced skew
            'greed': 0.08,              # Call skew emerging
            'extreme_greed': 0.15       # Strong call skew
        }
        
        # Multi-factor fear/greed scoring
        self.scoring_weights = {
            'put_call_skew': 0.40,      # Primary factor
            'iv_percentile': 0.25,       # Historical context
            'term_structure': 0.20,      # Time decay patterns
            'surface_curvature': 0.15    # Smile intensity
        }
    
    def analyze_fear_greed_sentiment(self, iv_analysis_results, market_data):
        """Analyze market fear/greed based on comprehensive IV analysis"""
        fear_greed_analysis = {}
        
        # Extract key IV metrics
        basic_metrics = iv_analysis_results['basic_skew_metrics']
        percentile_analysis = iv_analysis_results.get('iv_percentile_analysis', {})
        surface_analysis = iv_analysis_results.get('surface_analysis', {})
        
        # Calculate component scores
        skew_score = self.calculate_skew_fear_greed_score(basic_metrics)
        percentile_score = self.calculate_percentile_fear_greed_score(percentile_analysis)
        term_structure_score = self.calculate_term_structure_score(iv_analysis_results)
        surface_score = self.calculate_surface_curvature_score(surface_analysis)
        
        # Weighted composite score
        composite_score = (
            skew_score * self.scoring_weights['put_call_skew'] +
            percentile_score * self.scoring_weights['iv_percentile'] +
            term_structure_score * self.scoring_weights['term_structure'] +
            surface_score * self.scoring_weights['surface_curvature']
        )
        
        # Classify sentiment
        sentiment_classification = self.classify_fear_greed_sentiment(composite_score)
        
        fear_greed_analysis = {
            'component_scores': {
                'skew_score': skew_score,
                'percentile_score': percentile_score,
                'term_structure_score': term_structure_score,
                'surface_score': surface_score
            },
            'composite_score': composite_score,
            'sentiment_classification': sentiment_classification,
            'confidence_level': self.calculate_sentiment_confidence(
                skew_score, percentile_score, term_structure_score, surface_score
            ),
            'regime_implications': self.generate_sentiment_regime_implications(
                sentiment_classification, composite_score
            )
        }
        
        return fear_greed_analysis
```

---

## **7. Performance Optimization & Production Integration**

### **Performance Targets from Production Implementation**
```python
class ProductionPerformanceOptimizer:
    def __init__(self):
        # Performance targets from existing implementation
        self.performance_targets = {
            'iv_surface_analysis': 200,      # <200ms target
            'threshold_optimization': 30,    # <30ms target
            'sentiment_classification': 92,  # >92% accuracy target
            'memory_allocation': 600,        # <600MB additional memory
            'confidence_scoring': 90         # >90% confidence accuracy
        }
        
        # Memory optimization strategies
        self.memory_optimization = {
            'deque_based_history': True,     # Memory-efficient rolling windows
            'vectorized_calculations': True, # NumPy optimization
            'component_caching': True,       # Result caching strategy
            'parallel_processing': True      # Multi-component concurrent execution
        }
    
    def optimize_iv_analysis_performance(self, iv_analysis_components):
        """Optimize IV analysis performance for production"""
        optimization_results = {}
        
        # Memory optimization
        memory_optimized_components = self.apply_memory_optimization(
            iv_analysis_components
        )
        
        # Speed optimization
        speed_optimized_components = self.apply_speed_optimization(
            memory_optimized_components
        )
        
        # Accuracy optimization
        accuracy_optimized_components = self.apply_accuracy_optimization(
            speed_optimized_components
        )
        
        # Validate performance targets
        performance_validation = self.validate_performance_targets(
            accuracy_optimized_components
        )
        
        optimization_results = {
            'optimized_components': accuracy_optimized_components,
            'performance_validation': performance_validation,
            'optimization_gains': self.calculate_optimization_gains(
                iv_analysis_components, accuracy_optimized_components
            ),
            'production_readiness_score': self.calculate_production_readiness(
                performance_validation
            )
        }
        
        return optimization_results
```

The IV Skew Analysis System now represents a sophisticated, production-ready approach to volatility-based regime detection that leverages the existing comprehensive implementation while adding advanced adaptive learning and regime-specific optimizations.