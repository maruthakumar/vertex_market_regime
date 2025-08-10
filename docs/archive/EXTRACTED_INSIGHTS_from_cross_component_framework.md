# EXTRACTED INSIGHTS from Cross-Component Correlation Framework
## Unique Patterns and Thresholds Preserved Before Archiving

---

## **🎯 CRITICAL MARKET ALIGNMENT PATTERNS (UNIQUE INSIGHTS)**

### **Specific Bullish/Bearish Threshold Values**
```python
# These specific threshold values were defined in the original framework
Expected_Correlation_Patterns = {
    'bullish_alignment': {
        'otm_decay_threshold': -0.03,        # OTM should decay by 3%+
        'itm_strength_threshold': 0.02,      # ITM should strengthen by 2%+  
        'atm_pe_decay_threshold': -0.02,     # ATM PE should weaken by 2%+
        'atm_ce_strength_threshold': 0.01    # ATM CE should strengthen by 1%+
    },
    'bearish_alignment': {
        'itm_decay_threshold': -0.03,        # ITM should decay by 3%+
        'otm_strength_threshold': 0.02,      # OTM should strengthen by 2%+
        'atm_ce_decay_threshold': -0.02,     # ATM CE should weaken by 2%+  
        'atm_pe_strength_threshold': 0.01    # ATM PE should strengthen by 1%+
    },
    'sideways_alignment': {
        'all_straddles_stable': 0.005,       # All should be stable within 0.5%
        'theta_decay_dominance': -0.01       # Time decay should dominate at 1%
    }
}
```

### **Correlation Strength Classification**
```python
# Original correlation strength thresholds
Correlation_Thresholds = {
    'strong_correlation': 0.8,
    'moderate_correlation': 0.6, 
    'weak_correlation': 0.4,
    'non_correlation': 0.2,
    'divergence': 0.0
}

# Rolling correlation windows
Correlation_Windows = {
    'short_term': 20,      # 20-period correlation
    'medium_term': 50,     # 50-period correlation
    'long_term': 100       # 100-period correlation
}
```

### **Key Market Intelligence Patterns**
```python
# CRITICAL PATTERNS identified in original framework:
Market_Regime_Patterns = {
    
    # 1. BULLISH PATTERN
    'bullish_signature': 'OTM decay + ITM strengthening + ATM PE weakening',
    'bullish_logic': 'Market moving up = OTM calls lose premium, ITM calls gain, puts weaken',
    
    # 2. BEARISH PATTERN  
    'bearish_signature': 'ITM decay + OTM strengthening + ATM CE weakening',
    'bearish_logic': 'Market moving down = ITM calls lose premium, OTM puts gain, calls weaken',
    
    # 3. SIDEWAYS PATTERN
    'sideways_signature': 'All straddles stable + theta decay dominance',
    'sideways_logic': 'No direction = straddle prices stable, time decay primary factor'
}
```

### **Directional Weight Dynamics**
```python
# Call-side vs Put-side weight correlation patterns
Directional_Correlations = {
    'call_side_weight_increase': {
        'correlation_with': 'bullish_alignment',
        'expected_pattern': 'Call premiums increase, put premiums decrease'
    },
    'put_side_weight_increase': {
        'correlation_with': 'bearish_alignment', 
        'expected_pattern': 'Put premiums increase, call premiums decrease'
    },
    'weight_balance': {
        'correlation_with': 'sideways_alignment',
        'expected_pattern': 'Call and put weights remain balanced'
    }
}
```

---

## **📊 CROSS-COMPONENT CORRELATION EXPECTATIONS**

### **Expected Cross-Component Behavior**
```python
Expected_Cross_Correlations = {
    'component_1_straddle_vs_component_2_greeks': {
        'expected_correlation': 'HIGH (0.7-0.9)',
        'logic': 'Straddle prices should correlate with Greeks sentiment'
    },
    'component_1_straddle_vs_component_3_oi_pa': {
        'expected_correlation': 'MEDIUM-HIGH (0.6-0.8)',
        'logic': 'Straddle movement should correlate with institutional flow'
    },
    'component_2_greeks_vs_component_3_oi_pa': {
        'expected_correlation': 'MEDIUM (0.5-0.7)',
        'logic': 'Greeks sentiment should align with OI flow patterns'
    },
    'component_3_oi_pa_vs_component_7_support_resistance': {
        'expected_correlation': 'MEDIUM (0.4-0.6)',
        'logic': 'OI flow should respect support/resistance levels'
    }
}
```

### **Regime-Specific Expected Correlations**
```python
Regime_Specific_Correlations = {
    'LVLD': {  # Low Volatility Low Delta
        'all_correlations': 'HIGH (0.8+)',
        'reason': 'Stable market = high component agreement'
    },
    'HVC': {   # High Volatility Contraction  
        'correlations': 'MEDIUM-HIGH (0.6-0.8)',
        'reason': 'Volatility declining = moderate agreement'
    },
    'TBVE': {  # Trend Breaking Volatility Expansion
        'correlations': 'LOW-MEDIUM (0.3-0.6)',
        'reason': 'Trend change = component disagreement'
    },
    'PSED': {  # Poor Sentiment Elevated Divergence
        'correlations': 'LOW (0.2-0.4)',
        'reason': 'Poor sentiment = high component divergence'
    }
}
```

---

## **⚡ SPECIFIC ACTION FRAMEWORKS**

### **Correlation Scenario Actions**
```python
Correlation_Actions = {
    'high_correlation_scenario': {
        'confidence': 'HIGH',
        'action': 'Trust regime classification, proceed with high confidence',
        'risk': 'LOW'
    },
    'moderate_correlation_scenario': {
        'confidence': 'MEDIUM',
        'action': 'Validate with additional data, proceed with caution',
        'risk': 'MEDIUM'
    },
    'non_correlation_scenario': {
        'confidence': 'LOW', 
        'action': 'Wait for correlation restoration or regime change confirmation',
        'risk': 'HIGH'
    },
    'divergence_scenario': {
        'confidence': 'REGIME_CHANGE_LIKELY',
        'action': 'Prepare for regime transition, increase monitoring',
        'risk': 'VERY_HIGH'
    }
}
```

---

## **🎯 INTEGRATION NOTES FOR COMPONENT 6**

### **How These Insights Enhance Component 6:**

1. **Specific Threshold Values**: Component 6 can use these validated threshold values as starting points for historical learning
2. **Market Alignment Patterns**: The bullish/bearish/sideways patterns provide expert domain knowledge for ML training
3. **Correlation Strength Classification**: The 5-tier correlation classification (strong → divergence) provides structured interpretation
4. **Action Frameworks**: The specific actions for each correlation scenario provide decision logic structure

### **Recommendation**: 
Integrate these specific patterns and thresholds into Component 6's historical learning system as **expert priors** and **validation benchmarks**.

---

*Document Status: ARCHIVED - Insights extracted and integrated into Component 6*
*Date: $(date)*
*Reason: Superseded by Component 6 Ultra-Comprehensive Correlation Framework*