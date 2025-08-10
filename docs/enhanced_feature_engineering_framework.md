# Enhanced Feature Engineering Framework for 8-Regime Market Classification

## Executive Summary

This document presents the corrected and enhanced feature engineering framework for market regime detection, implementing a sophisticated 8-regime classification system with comprehensive rolling straddle analysis, Greeks sentiment analysis, trending OI with price action analysis, and IV skew analysis.

**Key Enhancements:**
- **8-Regime Classification System** (aligned with Advanced Option Premium Regime Framework)
- **Rolling Straddle Analysis** with symmetric straddle methodology
- **Enhanced Greeks Sentiment** with volume-weighted calculations
- **Trending OI with PA Analysis** with exact pattern replication
- **IV Skew Analysis** with 7-level sentiment classification
- **Configuration-Driven Architecture** with Excel-based parameter management

## Table of Contents

1. [8-Regime Classification Framework](#8-regime-classification)
2. [Rolling Straddle Analysis](#rolling-straddle-analysis)
3. [Greeks Sentiment Analysis](#greeks-sentiment-analysis)
4. [Trending OI with Price Action Analysis](#trending-oi-analysis)
5. [IV Skew Analysis](#iv-skew-analysis)
6. [Feature Engineering Pipeline](#feature-engineering-pipeline)
7. [Configuration Management](#configuration-management)
8. [Implementation Specifications](#implementation-specifications)

## 1. 8-Regime Classification Framework {#8-regime-classification}

Based on the Advanced Option Premium Regime Framework, we implement the following 8 distinct market regimes:

### Regime Definitions

#### **Regime 1: Low Volatility Linear Decay (LVLD)**
- **Characteristics**: Consistently low volatility, predictable theta decay
- **Volatility**: < 25th percentile
- **Term Structure**: Normal contango
- **Skew**: Within normal ranges
- **Target Win Rate**: >80%
- **Position Sizing**: Aggressive (1.2-1.5x base)

```python
LVLD_CRITERIA = {
    'realized_vol_percentile': '<25',
    'implied_vol_rank': '<30',
    'term_structure': 'contango',
    'volatility_clustering': '<0.3',
    'premium_decay_pattern': 'linear'
}
```

#### **Regime 2: High Volatility Clustering (HVC)**
- **Characteristics**: Elevated volatility with strong clustering effects
- **Volatility**: > 75th percentile with high persistence
- **Clustering Coefficient**: >0.7
- **Target Win Rate**: 60-70%
- **Position Sizing**: Reduced (0.7-0.8x base)

```python
HVC_CRITERIA = {
    'realized_vol_percentile': '>75',
    'volatility_clustering': '>0.7',
    'volatility_persistence': '>0.8',
    'implied_vol_expansion': 'elevated',
    'gamma_risk': 'high'
}
```

#### **Regime 3: Volatility Crush Post-Event (VCPE)**
- **Characteristics**: Rapid volatility contraction following events
- **Duration**: 2-5 trading days typically
- **IV Decline**: Rapid across all strikes
- **Target Win Rate**: 85-95%
- **Position Sizing**: Aggressive but time-limited

```python
VCPE_CRITERIA = {
    'iv_contraction_rate': '>20%_daily',
    'term_structure_normalization': 'rapid',
    'skew_compression': 'significant',
    'event_proximity': 'post_event',
    'theta_acceleration': 'non_linear'
}
```

#### **Regime 4: Trending Bull with Volatility Expansion (TBVE)**
- **Characteristics**: Positive momentum with expanding volatility
- **Price Action**: Sustained positive momentum
- **Volatility**: Increasing despite positive movement
- **Target Win Rate**: 70-80%
- **Strategy Focus**: Put-based strategies

```python
TBVE_CRITERIA = {
    'price_momentum': 'positive_sustained',
    'implied_vol_trend': 'expanding',
    'skew_steepening': 'put_protection_demand',
    'term_structure': 'flattening_or_inversion',
    'delta_vega_interaction': 'complex'
}
```

#### **Regime 5: Trending Bear with Volatility Spike (TBVS)**
- **Characteristics**: Negative momentum with explosive volatility
- **Risk Level**: Highest for option sellers
- **Volatility**: Extreme spikes, particularly in puts
- **Target Win Rate**: <50%
- **Position Sizing**: Minimal (0.3-0.5x base)

```python
TBVS_CRITERIA = {
    'price_momentum': 'negative_sustained',
    'implied_vol_spikes': 'explosive',
    'put_skew': 'extreme',
    'term_structure': 'inverted',
    'tail_risk_premium': 'elevated'
}
```

#### **Regime 6: Sideways Choppy with Gamma Scalping (SCGS)**
- **Characteristics**: Range-bound with high volatility and gamma exposure
- **Price Action**: Well-defined support/resistance
- **Opportunity**: Gamma scalping and iron condors
- **Target Win Rate**: 65-75%
- **Strategy**: Dynamic gamma hedging

```python
SCGS_CRITERIA = {
    'price_range_bound': 'defined_levels',
    'realized_vol': 'high_despite_limited_direction',
    'gamma_exposure': 'significant',
    'reversal_frequency': 'high',
    'whipsaw_risk': 'elevated'
}
```

#### **Regime 7: Premium Spike Event-Driven (PSED)**
- **Characteristics**: Scheduled event-driven premium expansion
- **Events**: Earnings, FDA approvals, economic releases
- **Duration**: Predictable timeline
- **Target Win Rate**: >80% with timing
- **Strategy**: Event-based volatility selling

```python
PSED_CRITERIA = {
    'scheduled_event': 'upcoming',
    'implied_vol_expansion': 'pre_event',
    'premium_inflation': 'across_strikes',
    'event_timeline': 'predictable',
    'post_event_crush': 'expected'
}
```

#### **Regime 8: Correlation Breakdown Volatility (CBV)**
- **Characteristics**: Traditional correlations breaking down
- **Cross-Asset Effects**: Spillover volatility
- **Complexity**: Multi-asset analysis required
- **Target Win Rate**: Variable
- **Strategy**: Relative value and cross-asset hedging

```python
CBV_CRITERIA = {
    'correlation_stability': 'breaking_down',
    'cross_asset_spillovers': 'unusual_patterns',
    'traditional_hedging': 'ineffective',
    'volatility_source': 'structural_shifts',
    'regime_transition': 'major_market_change'
}
```

### Regime Detection Algorithm

```python
class EightRegimeClassifier:
    """8-Regime classification system implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_weights = {
            'volatility_features': 0.30,
            'straddle_analysis': 0.25,
            'greeks_sentiment': 0.20,
            'oi_price_action': 0.15,
            'iv_skew': 0.10
        }
    
    def classify_regime(self, features: Dict[str, Any]) -> str:
        """Classify market regime based on engineered features"""
        
        # Calculate regime scores
        regime_scores = {}
        
        for regime in ['LVLD', 'HVC', 'VCPE', 'TBVE', 'TBVS', 'SCGS', 'PSED', 'CBV']:
            score = self._calculate_regime_score(regime, features)
            regime_scores[regime] = score
        
        # Find best matching regime
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        confidence = self._calculate_confidence(regime_scores, best_regime[0])
        
        return {
            'regime': best_regime[0],
            'confidence': confidence,
            'all_scores': regime_scores
        }
    
    def _calculate_regime_score(self, regime: str, features: Dict[str, Any]) -> float:
        """Calculate score for specific regime"""
        criteria = getattr(self, f"{regime}_CRITERIA")
        score = 0.0
        
        for criterion, weight in self.feature_weights.items():
            criterion_score = self._evaluate_criterion(criterion, features, criteria)
            score += criterion_score * weight
        
        return score
```

## 2. Rolling Straddle Analysis {#rolling-straddle-analysis}

### Enhanced Triple Straddle System

Building on your existing framework, the rolling straddle analysis implements a sophisticated 10-component system:

#### **Component Structure**

```python
STRADDLE_COMPONENTS = {
    # Individual Options (6 components)
    'atm_ce': {'weight': 0.08, 'analysis': 'Call directional bias'},
    'atm_pe': {'weight': 0.08, 'analysis': 'Put fear gauge'},
    'itm_ce': {'weight': 0.06, 'analysis': 'Deep call value'},
    'itm_pe': {'weight': 0.06, 'analysis': 'Deep put value'},
    'otm_ce': {'weight': 0.04, 'analysis': 'Speculative call activity'},
    'otm_pe': {'weight': 0.04, 'analysis': 'Tail risk hedging'},
    
    # Symmetric Straddles (3 components)
    'atm_straddle': {'weight': 0.20, 'analysis': 'Pure volatility play'},
    'itm1_straddle': {'weight': 0.15, 'analysis': 'Delta-heavy straddle'},
    'otm1_straddle': {'weight': 0.10, 'analysis': 'Vega-heavy straddle'},
    
    # Combined Triple Straddle (1 component)
    'combined_straddle': {'weight': 0.19, 'analysis': '50/30/20 weighted combination'}
}
```

#### **Rolling Analysis Framework**

```python
class RollingStraddleAnalyzer:
    """Rolling straddle analysis with symmetric methodology"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeframes = ['3min', '5min', '10min', '15min']
        self.rolling_windows = {
            '3min': 20,   # 1 hour
            '5min': 36,   # 3 hours
            '10min': 36,  # 6 hours
            '15min': 32   # 8 hours
        }
    
    def analyze_straddle_components(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze all 10 straddle components"""
        
        results = {}
        
        # Extract components from 49-column dataset
        components = self._extract_components(data)
        
        # Rolling analysis for each timeframe
        for timeframe in self.timeframes:
            timeframe_results = {}
            window = self.rolling_windows[timeframe]
            
            for component_name, component_data in components.items():
                # Technical analysis
                technical_score = self._analyze_component_technicals(
                    component_data, window, timeframe
                )
                
                # Rolling statistics
                rolling_stats = self._calculate_rolling_statistics(
                    component_data, window
                )
                
                timeframe_results[component_name] = {
                    'technical_score': technical_score,
                    'rolling_stats': rolling_stats,
                    'weight': STRADDLE_COMPONENTS[component_name]['weight']
                }
            
            results[timeframe] = timeframe_results
        
        # Calculate 10x10 correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(components)
        
        # Generate combined analysis
        combined_analysis = self._generate_combined_analysis(results, correlation_matrix)
        
        return {
            'component_analysis': results,
            'correlation_matrix': correlation_matrix,
            'combined_analysis': combined_analysis
        }
    
    def _extract_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Extract 10 components from 49-column dataset"""
        
        components = {}
        
        # Filter for different strike types
        atm_data = data[data['strike'] == data['atm_strike']]
        itm_data = data[data['call_strike_type'] == 'ITM1']
        otm_data = data[data['call_strike_type'] == 'OTM1']
        
        # Individual Options (6 components)
        components['atm_ce'] = atm_data['ce_close']
        components['atm_pe'] = atm_data['pe_close']
        components['itm_ce'] = itm_data['ce_close']
        components['itm_pe'] = itm_data['pe_close']
        components['otm_ce'] = otm_data['ce_close']
        components['otm_pe'] = otm_data['pe_close']
        
        # Symmetric Straddles (3 components)
        components['atm_straddle'] = atm_data['ce_close'] + atm_data['pe_close']
        components['itm1_straddle'] = itm_data['ce_close'] + itm_data['pe_close']
        components['otm1_straddle'] = otm_data['ce_close'] + otm_data['pe_close']
        
        # Combined Triple Straddle (1 component)
        components['combined_straddle'] = (
            0.50 * components['atm_straddle'] +
            0.30 * components['itm1_straddle'] +
            0.20 * components['otm1_straddle']
        )
        
        return components
```

### Rolling Analysis Features

#### **Technical Indicators per Component**
- **EMA Analysis**: 20/100/200 period alignments
- **VWAP Positioning**: Current vs VWAP relationship
- **Pivot Point Analysis**: Support/resistance levels
- **Momentum Indicators**: RSI, MACD for each component

#### **Rolling Statistics**
- **Volatility**: Rolling standard deviation
- **Skewness**: Distribution asymmetry
- **Kurtosis**: Tail risk measurement
- **Autocorrelation**: Persistence analysis

#### **Cross-Component Analysis**
- **10x10 Correlation Matrix**: Dynamic correlations
- **Component Divergence**: Unusual relationships
- **Regime Transition Signals**: Early warning system

## 3. Greeks Sentiment Analysis {#greeks-sentiment-analysis}

### Volume-Weighted Greeks Framework

```python
class GreeksSentimentAnalyzer:
    """Enhanced Greeks sentiment with volume weighting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.greek_weights = {
            'delta': 0.25,
            'gamma': 0.30,
            'theta': 0.20,
            'vega': 0.20,
            'rho': 0.05
        }
    
    def analyze_greeks_sentiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive Greeks sentiment analysis"""
        
        # Calculate volume-weighted Greeks
        vw_greeks = self._calculate_volume_weighted_greeks(data)
        
        # Greeks sentiment classification
        sentiment_scores = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            # Individual Greek analysis
            greek_analysis = self._analyze_individual_greek(data, greek)
            
            # Cross-Greek correlations
            correlations = self._analyze_greek_correlations(data, greek)
            
            # Time decay impact (for theta-sensitive analysis)
            time_impact = self._analyze_time_decay_impact(data, greek)
            
            sentiment_scores[greek] = {
                'individual_analysis': greek_analysis,
                'correlations': correlations,
                'time_impact': time_impact,
                'volume_weighted_value': vw_greeks[greek],
                'weight': self.greek_weights[greek]
            }
        
        # Combined Greeks sentiment
        combined_sentiment = self._calculate_combined_sentiment(sentiment_scores)
        
        # DTE-specific adjustments
        dte_adjustments = self._calculate_dte_adjustments(data, sentiment_scores)
        
        return {
            'individual_greeks': sentiment_scores,
            'combined_sentiment': combined_sentiment,
            'dte_adjustments': dte_adjustments,
            'volume_weighted_greeks': vw_greeks
        }
    
    def _calculate_volume_weighted_greeks(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volume-weighted Greeks"""
        
        vw_greeks = {}
        
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            # Call Greeks
            ce_greek_col = f'ce_{greek}'
            ce_volume_col = 'ce_volume'
            
            # Put Greeks  
            pe_greek_col = f'pe_{greek}'
            pe_volume_col = 'pe_volume'
            
            # Volume-weighted calculation
            ce_vw = (data[ce_greek_col] * data[ce_volume_col]).sum() / data[ce_volume_col].sum()
            pe_vw = (data[pe_greek_col] * data[pe_volume_col]).sum() / data[pe_volume_col].sum()
            
            # Combined volume-weighted Greek
            total_volume = data[ce_volume_col].sum() + data[pe_volume_col].sum()
            vw_greeks[greek] = (
                (ce_vw * data[ce_volume_col].sum() + pe_vw * data[pe_volume_col].sum()) / 
                total_volume
            )
        
        return vw_greeks
```

### Greeks Sentiment Classification

#### **7-Level Sentiment System**
```python
GREEKS_SENTIMENT_LEVELS = {
    'EXTREMELY_BULLISH': 3.0,
    'VERY_BULLISH': 2.0,
    'MODERATELY_BULLISH': 1.0,
    'NEUTRAL': 0.0,
    'MODERATELY_BEARISH': -1.0,
    'VERY_BEARISH': -2.0,
    'EXTREMELY_BEARISH': -3.0
}
```

#### **Greek-Specific Analysis**

**Delta Sentiment:**
- Measures directional bias
- Volume-weighted call vs put delta
- ATM delta positioning

**Gamma Sentiment:**
- Gamma exposure analysis
- ATM gamma concentration
- Pin risk assessment

**Theta Sentiment:**
- Time decay impact
- DTE-specific theta analysis
- Acceleration patterns

**Vega Sentiment:**
- Volatility sensitivity
- IV rank positioning
- Term structure impact

**Rho Sentiment:**
- Interest rate sensitivity
- Carry trade implications
- Long-term positioning

## 4. Trending OI with Price Action Analysis {#trending-oi-analysis}

### Enhanced OI Pattern Recognition

Building on your exact pattern replication system:

```python
class TrendingOIAnalyzer:
    """Enhanced trending OI with price action analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_signals = {
            # Bullish Patterns
            'long_build_up': 0.7,      # OI↑ + Price↑
            'short_covering': 0.6,     # OI↓ + Price↑
            'strong_bullish': 1.0,
            'mild_bullish': 0.5,
            'sideways_to_bullish': 0.2,
            
            # Bearish Patterns
            'short_build_up': -0.7,    # OI↑ + Price↓
            'long_unwinding': -0.6,    # OI↓ + Price↓
            'strong_bearish': -1.0,
            'mild_bearish': -0.5,
            'sideways_to_bearish': -0.2,
            
            # Neutral Patterns
            'neutral': 0.0,
            'sideways': 0.0
        }
    
    def analyze_oi_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive OI pattern analysis"""
        
        # Calculate OI changes
        oi_changes = self._calculate_oi_changes(data)
        
        # Price action analysis
        price_action = self._analyze_price_action(data)
        
        # Pattern classification
        patterns = self._classify_oi_patterns(oi_changes, price_action)
        
        # Time-of-day weight adjustments
        tod_weights = self._calculate_time_of_day_weights(data)
        
        # Multi-timeframe analysis
        mtf_analysis = self._multi_timeframe_analysis(data)
        
        # Institutional flow detection
        institutional_flow = self._detect_institutional_flow(data)
        
        return {
            'oi_changes': oi_changes,
            'price_action': price_action,
            'patterns': patterns,
            'time_of_day_weights': tod_weights,
            'multi_timeframe': mtf_analysis,
            'institutional_flow': institutional_flow
        }
    
    def _calculate_oi_changes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate OI changes for calls and puts"""
        
        # Call OI analysis
        ce_oi_change = data['ce_oi'].pct_change()
        ce_coi_change = data['ce_coi'].pct_change()
        
        # Put OI analysis
        pe_oi_change = data['pe_oi'].pct_change()
        pe_coi_change = data['pe_coi'].pct_change()
        
        # Combined OI metrics
        total_oi = data['ce_oi'] + data['pe_oi']
        put_call_oi_ratio = data['pe_oi'] / data['ce_oi']
        
        return {
            'ce_oi_change': ce_oi_change,
            'ce_coi_change': ce_coi_change,
            'pe_oi_change': pe_oi_change,
            'pe_coi_change': pe_coi_change,
            'total_oi': total_oi,
            'put_call_oi_ratio': put_call_oi_ratio
        }
```

### OI Pattern Features

#### **Primary Patterns**
- **Long Build-up**: OI↑ + Price↑ (Bullish)
- **Short Build-up**: OI↑ + Price↓ (Bearish)
- **Long Unwinding**: OI↓ + Price↓ (Bearish)
- **Short Covering**: OI↓ + Price↑ (Bullish)

#### **Advanced Features**
- **Put-Call OI Ratio**: Sentiment indicator
- **OI Concentration**: Strike-wise OI distribution
- **Institutional Detection**: Large block analysis
- **Cross-Strike Analysis**: OI migration patterns

## 5. IV Skew Analysis {#iv-skew-analysis}

### 7-Level IV Skew Sentiment

```python
class IVSkewAnalyzer:
    """Advanced IV skew analysis with 7-level classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.skew_thresholds = {
            'EXTREMELY_BEARISH': -0.15,   # Very high put skew
            'VERY_BEARISH': -0.10,
            'MODERATELY_BEARISH': -0.05,
            'NEUTRAL': 0.00,
            'MODERATELY_BULLISH': 0.05,
            'VERY_BULLISH': 0.10,
            'EXTREMELY_BULLISH': 0.15     # Very high call skew
        }
    
    def analyze_iv_skew(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive IV skew analysis"""
        
        # Put-Call IV skew
        put_call_skew = self._calculate_put_call_skew(data)
        
        # Strike-based skew profile
        strike_skew = self._calculate_strike_skew_profile(data)
        
        # Term structure skew
        term_skew = self._calculate_term_structure_skew(data)
        
        # Skew sentiment classification
        skew_sentiment = self._classify_skew_sentiment(put_call_skew)
        
        # Skew regime detection
        skew_regime = self._detect_skew_regime(data)
        
        # Confidence scoring
        confidence = self._calculate_skew_confidence(data)
        
        return {
            'put_call_skew': put_call_skew,
            'strike_skew_profile': strike_skew,
            'term_structure_skew': term_skew,
            'skew_sentiment': skew_sentiment,
            'skew_regime': skew_regime,
            'confidence': confidence
        }
    
    def _calculate_put_call_skew(self, data: pd.DataFrame) -> float:
        """Calculate ATM put-call IV skew"""
        
        atm_data = data[data['strike'] == data['atm_strike']]
        
        if len(atm_data) == 0:
            return 0.0
        
        ce_iv = atm_data['ce_iv'].iloc[-1]
        pe_iv = atm_data['pe_iv'].iloc[-1]
        
        # Skew = Put IV - Call IV
        skew = pe_iv - ce_iv
        
        return skew
```

### IV Skew Features

#### **Skew Metrics**
- **ATM Put-Call Skew**: Primary sentiment indicator
- **Strike Distribution**: OTM put vs call premiums
- **Term Structure**: Skew across expirations
- **Skew Smile**: Full volatility surface analysis

#### **Regime Integration**
- **Skew Regimes**: Normal, stressed, euphoric
- **Tail Risk Pricing**: Black swan protection
- **Cross-Asset Skew**: Correlation with other assets

## 6. Feature Engineering Pipeline {#feature-engineering-pipeline}

### Comprehensive Feature Set

```python
class FeatureEngineeringPipeline:
    """Complete feature engineering for 8-regime classification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_categories = {
            'volatility_features': 15,
            'straddle_features': 25,
            'greeks_features': 20,
            'oi_features': 12,
            'skew_features': 8,
            'technical_features': 20
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features for regime classification"""
        
        features = pd.DataFrame(index=data.index)
        
        # 1. Volatility Features (15 features)
        vol_features = self._engineer_volatility_features(data)
        features = pd.concat([features, vol_features], axis=1)
        
        # 2. Straddle Features (25 features)
        straddle_features = self._engineer_straddle_features(data)
        features = pd.concat([features, straddle_features], axis=1)
        
        # 3. Greeks Features (20 features)
        greeks_features = self._engineer_greeks_features(data)
        features = pd.concat([features, greeks_features], axis=1)
        
        # 4. OI Features (12 features)
        oi_features = self._engineer_oi_features(data)
        features = pd.concat([features, oi_features], axis=1)
        
        # 5. Skew Features (8 features)
        skew_features = self._engineer_skew_features(data)
        features = pd.concat([features, skew_features], axis=1)
        
        # 6. Technical Features (20 features)
        technical_features = self._engineer_technical_features(data)
        features = pd.concat([features, technical_features], axis=1)
        
        return features
    
    def _engineer_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer volatility-related features"""
        
        features = pd.DataFrame(index=data.index)
        
        # Realized volatility
        returns = data['spot'].pct_change()
        features['realized_vol_5d'] = returns.rolling(5).std() * np.sqrt(252)
        features['realized_vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
        features['realized_vol_60d'] = returns.rolling(60).std() * np.sqrt(252)
        
        # IV metrics
        features['atm_iv'] = data[data['strike'] == data['atm_strike']]['ce_iv']
        features['iv_rank_20d'] = features['atm_iv'].rolling(20).rank() / 20
        features['iv_percentile_60d'] = features['atm_iv'].rolling(60).rank() / 60
        
        # Volatility clustering
        features['vol_clustering'] = self._calculate_volatility_clustering(returns)
        
        # Term structure
        features['vol_term_structure'] = self._calculate_term_structure(data)
        
        return features
```

### Feature Categories

#### **1. Volatility Features (15 total)**
- Realized volatility (multiple timeframes)
- Implied volatility metrics
- Volatility clustering coefficients
- Term structure slopes
- Volatility regime indicators

#### **2. Straddle Features (25 total)**
- 10 component prices
- 10 component technical scores
- 3 correlation strength measures
- 2 combined analysis scores

#### **3. Greeks Features (20 total)**
- 5 volume-weighted Greeks
- 5 Greeks sentiment scores
- 5 cross-Greek correlations
- 5 DTE-adjusted Greeks

#### **4. OI Features (12 total)**
- OI change patterns
- Put-call OI ratios
- Institutional flow indicators
- Strike distribution metrics

#### **5. Skew Features (8 total)**
- Put-call skew
- Strike skew profile
- Term structure skew
- Skew regime indicators

#### **6. Technical Features (20 total)**
- Price momentum
- Support/resistance levels
- Volume analysis
- Multi-timeframe alignment

## 7. Configuration Management {#configuration-management}

### Excel Configuration Structure

The system uses Excel-based configuration with the following structure:

#### **MR_CONFIG_8REGIME_1.0.0.xlsx**

**Sheet: Regime_Parameters**
| Regime | Vol_Threshold | Clustering | Term_Structure | Skew_Range | Position_Sizing |
|--------|---------------|------------|----------------|------------|-----------------|
| LVLD   | <25th pct     | <0.3       | Contango       | Normal     | 1.2-1.5x       |
| HVC    | >75th pct     | >0.7       | Any            | Any        | 0.7-0.8x       |
| VCPE   | Contracting   | Any        | Normalizing    | Compressing| 1.0-1.3x       |
| TBVE   | Expanding     | Any        | Flattening     | Steepening | 1.0x           |
| TBVS   | Spiking       | >0.8       | Inverted       | Extreme    | 0.3-0.5x       |
| SCGS   | High          | Moderate   | Any            | Any        | 0.8-1.0x       |
| PSED   | Pre-Event     | Any        | Any            | Event-driven| 1.0-1.2x      |
| CBV    | Unusual       | Breakdown  | Distorted      | Unstable   | 0.6-0.9x       |

**Sheet: Straddle_Configuration**
| Component | Weight | Technical_Window | Rolling_Window | Correlation_Threshold |
|-----------|--------|------------------|----------------|-----------------------|
| ATM_CE    | 0.08   | 20               | 50             | 0.7                   |
| ATM_PE    | 0.08   | 20               | 50             | 0.7                   |
| ITM_CE    | 0.06   | 20               | 50             | 0.6                   |
| ITM_PE    | 0.06   | 20               | 50             | 0.6                   |
| OTM_CE    | 0.04   | 20               | 50             | 0.5                   |
| OTM_PE    | 0.04   | 20               | 50             | 0.5                   |
| ATM_Straddle | 0.20 | 20               | 50             | 0.8                   |
| ITM1_Straddle| 0.15 | 20               | 50             | 0.7                   |
| OTM1_Straddle| 0.10 | 20               | 50             | 0.6                   |
| Combined_Straddle| 0.19 | 20           | 50             | 0.9                   |

**Sheet: Greeks_Parameters**
| Greek | Weight | Volume_Weight | DTE_Adjustment | Sentiment_Threshold |
|-------|--------|---------------|----------------|---------------------|
| Delta | 0.25   | True          | Linear         | 0.6                 |
| Gamma | 0.30   | True          | Exponential    | 0.7                 |
| Theta | 0.20   | True          | Accelerating   | 0.5                 |
| Vega  | 0.20   | True          | IV_Based       | 0.6                 |
| Rho   | 0.05   | False         | Interest_Rate  | 0.3                 |

**Sheet: OI_Pattern_Config**
| Pattern | Signal_Strength | Time_Weight | Institutional_Threshold |
|---------|-----------------|-------------|-------------------------|
| Long_Build_Up | 0.7 | Morning: 1.2 | 10000                   |
| Short_Build_Up | -0.7 | Morning: 1.2 | 10000                  |
| Long_Unwinding | -0.6 | Afternoon: 1.1 | 5000                 |
| Short_Covering | 0.6 | Afternoon: 1.1 | 5000                  |

**Sheet: IV_Skew_Config**
| Skew_Level | Threshold | Regime_Impact | Confidence_Weight |
|------------|-----------|---------------|-------------------|
| Extremely_Bearish | <-0.15 | 0.3 | 0.9 |
| Very_Bearish | -0.10 | 0.2 | 0.8 |
| Moderately_Bearish | -0.05 | 0.1 | 0.7 |
| Neutral | 0.00 | 0.0 | 0.6 |
| Moderately_Bullish | 0.05 | -0.1 | 0.7 |
| Very_Bullish | 0.10 | -0.2 | 0.8 |
| Extremely_Bullish | >0.15 | -0.3 | 0.9 |

## 8. Implementation Specifications {#implementation-specifications}

### System Architecture

```python
class EnhancedMarketRegimeSystem:
    """Complete 8-regime market classification system"""
    
    def __init__(self, config_path: str):
        self.config = self._load_excel_config(config_path)
        self.feature_pipeline = FeatureEngineeringPipeline(self.config)
        self.regime_classifier = EightRegimeClassifier(self.config)
        self.straddle_analyzer = RollingStraddleAnalyzer(self.config)
        self.greeks_analyzer = GreeksSentimentAnalyzer(self.config)
        self.oi_analyzer = TrendingOIAnalyzer(self.config)
        self.skew_analyzer = IVSkewAnalyzer(self.config)
    
    def process_market_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Complete market regime analysis"""
        
        # Feature engineering
        features = self.feature_pipeline.engineer_features(data)
        
        # Component analysis
        straddle_analysis = self.straddle_analyzer.analyze_straddle_components(data)
        greeks_analysis = self.greeks_analyzer.analyze_greeks_sentiment(data)
        oi_analysis = self.oi_analyzer.analyze_oi_patterns(data)
        skew_analysis = self.skew_analyzer.analyze_iv_skew(data)
        
        # Regime classification
        regime_result = self.regime_classifier.classify_regime({
            'features': features,
            'straddle': straddle_analysis,
            'greeks': greeks_analysis,
            'oi': oi_analysis,
            'skew': skew_analysis
        })
        
        return {
            'regime': regime_result,
            'components': {
                'straddle': straddle_analysis,
                'greeks': greeks_analysis,
                'oi': oi_analysis,
                'skew': skew_analysis
            },
            'features': features
        }
```

### Performance Requirements

- **Processing Time**: <3 seconds per analysis
- **Regime Accuracy**: >90% classification accuracy
- **Feature Completeness**: 100 engineered features
- **Configuration Flexibility**: Excel-based hot-reload
- **Scalability**: Handle multiple symbols simultaneously

### Integration Points

- **HeavyDB Integration**: Direct database connectivity
- **Vertex AI Deployment**: Cloud-based model serving
- **Real-time Processing**: Streaming data pipeline
- **Configuration Management**: Excel-based parameters
- **Monitoring**: Performance and accuracy tracking

This enhanced feature engineering framework provides the foundation for sophisticated market regime detection with comprehensive analysis of all key options market components.