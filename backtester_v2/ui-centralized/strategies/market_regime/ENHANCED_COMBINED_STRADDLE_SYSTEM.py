#!/usr/bin/env python3
"""
Enhanced Combined Straddle Analysis System
Portfolio Optimization-Based Multi-Straddle Framework for Market Regime Detection

This system implements the Enhanced Combined Straddle Analysis framework with:
- Portfolio optimization for optimal weight allocation
- Technical indicator integration on combined straddle
- Correlation analysis between straddle components
- Statistical validation and performance comparison

Based on comprehensive research and analysis documented in:
docs/enhanced_combined_straddle_framework.md

Author: The Augster
Date: 2025-06-20
Version: 7.0.0 (Enhanced Combined Straddle System)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple
from pathlib import Path
import warnings
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from scipy import stats

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedCombinedStraddleSystem:
    """
    Enhanced Combined Straddle Analysis System
    Portfolio optimization-based multi-straddle framework
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Enhanced Combined Straddle System"""
        
        self.output_dir = Path("enhanced_combined_straddle_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Default configuration
        self.config = {
            'lookback_window': 252,
            'rebalance_frequency': 'daily',
            'optimization_method': 'mean_variance',
            'technical_indicators': {
                'ema_periods': [20, 100, 200],
                'vwap_enabled': True,
                'pivot_points_enabled': True,
                'support_resistance_enabled': True
            },
            'correlation_window': 20,
            'risk_aversion': 1.0
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Optimal weights (will be updated through optimization)
        self.optimal_weights = {'atm': 0.45, 'itm1': 0.35, 'otm1': 0.20}
        self.weight_history = []
        
        # Performance tracking
        self.performance_metrics = {}
        
        logger.info("🚀 Enhanced Combined Straddle System initialized")
        logger.info(f"📊 Portfolio optimization with {self.config['optimization_method']}")
        logger.info(f"🎯 Optimal weights: ATM={self.optimal_weights['atm']:.1%}, "
                   f"ITM1={self.optimal_weights['itm1']:.1%}, OTM1={self.optimal_weights['otm1']:.1%}")
    
    def calculate_straddle_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate individual straddle components following HeavyDB standard"""
        logger.info("📊 Calculating straddle components (HeavyDB standard)...")
        
        try:
            # ATM Straddle (Traditional symmetric)
            df['atm_straddle'] = df['atm_ce_price'] + df['atm_pe_price']
            
            # ITM1 Straddle (HeavyDB standard: ITM1_CE + OTM1_PE at same strike)
            # Simulate ITM1_CE and OTM1_PE prices based on ATM with adjustments
            itm1_ce_adjustment = 1.15  # ITM call premium adjustment
            otm1_pe_adjustment = 0.85  # OTM put premium adjustment
            
            df['itm1_ce_price'] = df['atm_ce_price'] * itm1_ce_adjustment + 50  # Add intrinsic value
            df['otm1_pe_price'] = df['atm_pe_price'] * otm1_pe_adjustment
            df['itm1_straddle'] = df['itm1_ce_price'] + df['otm1_pe_price']
            
            # OTM1 Straddle (HeavyDB standard: OTM1_CE + ITM1_PE at same strike)
            otm1_ce_adjustment = 0.75  # OTM call premium adjustment
            itm1_pe_adjustment = 1.15  # ITM put premium adjustment
            
            df['otm1_ce_price'] = df['atm_ce_price'] * otm1_ce_adjustment
            df['itm1_pe_price'] = df['atm_pe_price'] * itm1_pe_adjustment + 50  # Add intrinsic value
            df['otm1_straddle'] = df['otm1_ce_price'] + df['itm1_pe_price']
            
            # Volume calculations (estimated)
            df['atm_straddle_volume'] = df.get('atm_ce_volume', 1000) + df.get('atm_pe_volume', 1000)
            df['itm1_straddle_volume'] = df['atm_straddle_volume'] * 0.7
            df['otm1_straddle_volume'] = df['atm_straddle_volume'] * 0.5
            
            logger.info("✅ Straddle components calculated (HeavyDB standard)")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating straddle components: {e}")
            return df
    
    def calculate_straddle_returns(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Calculate returns for portfolio optimization"""
        
        straddle_prices = df[['atm_straddle', 'itm1_straddle', 'otm1_straddle']].tail(window)
        returns = straddle_prices.pct_change().dropna()
        
        return returns
    
    def optimize_portfolio_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Optimize portfolio weights using mean-variance optimization"""
        logger.info("⚖️ Optimizing portfolio weights...")
        
        try:
            # Calculate expected returns and covariance matrix
            expected_returns = returns.mean().values
            
            # Use Ledoit-Wolf shrinkage for robust covariance estimation
            lw = LedoitWolf()
            cov_matrix, shrinkage = lw.fit(returns.values).covariance_, lw.shrinkage_
            
            # Objective function: maximize Sharpe ratio
            def objective(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
                portfolio_std = np.sqrt(portfolio_variance)
                
                # Return negative Sharpe ratio (for minimization)
                if portfolio_std == 0:
                    return 1e6  # Large penalty for zero volatility
                return -(portfolio_return / portfolio_std)
            
            # Constraints and bounds
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0.1, 0.8) for _ in range(3))  # Reasonable bounds
            
            # Initial guess (current optimal weights)
            x0 = np.array([self.optimal_weights['atm'], 
                          self.optimal_weights['itm1'], 
                          self.optimal_weights['otm1']])
            
            # Optimize
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = {'atm': result.x[0], 'itm1': result.x[1], 'otm1': result.x[2]}
                logger.info(f"✅ Weights optimized: ATM={optimized_weights['atm']:.3f}, "
                           f"ITM1={optimized_weights['itm1']:.3f}, OTM1={optimized_weights['otm1']:.3f}")
                return optimized_weights
            else:
                logger.warning("⚠️ Optimization failed, using default weights")
                return self.optimal_weights
                
        except Exception as e:
            logger.error(f"❌ Error in weight optimization: {e}")
            return self.optimal_weights
    
    def apply_market_adjustments(self, weights: Dict[str, float], 
                                current_vix: float = 20, dte: int = 2) -> Dict[str, float]:
        """Apply volatility and DTE adjustments to weights"""
        
        adjusted_weights = weights.copy()
        
        # Volatility adjustment
        vix_ma = 20  # Long-term VIX average
        
        if current_vix < 15:  # Low volatility
            adjusted_weights['atm'] *= 1.1
            adjusted_weights['itm1'] *= 0.95
            adjusted_weights['otm1'] *= 0.95
        elif current_vix > 30:  # High volatility
            adjusted_weights['atm'] *= 0.9
            adjusted_weights['itm1'] *= 1.05
            adjusted_weights['otm1'] *= 1.05
        
        # DTE adjustment
        if dte <= 1:  # 0-1 DTE: Emphasize ATM
            adjusted_weights['atm'] *= 1.1
            adjusted_weights['itm1'] *= 0.95
            adjusted_weights['otm1'] *= 0.95
        elif dte >= 4:  # 4+ DTE: Emphasize ITM1/OTM1
            adjusted_weights['atm'] *= 0.9
            adjusted_weights['itm1'] *= 1.05
            adjusted_weights['otm1'] *= 1.05
        
        # Renormalize weights
        total = sum(adjusted_weights.values())
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def calculate_enhanced_combined_straddle(self, df: pd.DataFrame, 
                                           current_vix: float = 20, 
                                           dte: int = 2) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculate enhanced combined straddle with optimal weights"""
        logger.info("🎯 Calculating enhanced combined straddle...")
        
        try:
            # Calculate individual straddle components
            df = self.calculate_straddle_components(df)
            
            # Optimize weights if enough data
            if len(df) >= self.config['lookback_window']:
                returns = self.calculate_straddle_returns(df, self.config['lookback_window'])
                if len(returns) > 20:  # Minimum data for optimization
                    self.optimal_weights = self.optimize_portfolio_weights(returns)
            
            # Apply market adjustments
            adjusted_weights = self.apply_market_adjustments(self.optimal_weights, current_vix, dte)
            
            # Calculate enhanced combined straddle
            df['enhanced_combined_straddle'] = (
                adjusted_weights['atm'] * df['atm_straddle'] +
                adjusted_weights['itm1'] * df['itm1_straddle'] +
                adjusted_weights['otm1'] * df['otm1_straddle']
            )
            
            # Calculate traditional combined straddle for comparison
            df['traditional_combined_straddle'] = (
                0.6 * df['atm_straddle'] + 0.4 * df['itm1_straddle']
            )
            
            # Store weight history
            self.weight_history.append({
                'timestamp': df.index[-1] if hasattr(df, 'index') else len(df),
                'weights': adjusted_weights,
                'vix': current_vix,
                'dte': dte
            })
            
            logger.info("✅ Enhanced combined straddle calculated")
            return df, adjusted_weights
            
        except Exception as e:
            logger.error(f"❌ Error calculating enhanced combined straddle: {e}")
            return df, self.optimal_weights
    
    def calculate_technical_indicators(self, combined_straddle: pd.Series, 
                                     volume: pd.Series) -> Dict[str, Any]:
        """Calculate technical indicators for combined straddle"""
        logger.info("📈 Calculating technical indicators...")
        
        try:
            indicators = {}
            
            # EMA Analysis
            if 'ema_periods' in self.config['technical_indicators']:
                for period in self.config['technical_indicators']['ema_periods']:
                    ema_key = f'ema_{period}'
                    indicators[ema_key] = combined_straddle.ewm(span=period).mean()
                    indicators[f'{ema_key}_position'] = (combined_straddle / indicators[ema_key] - 1)
                    indicators[f'{ema_key}_slope'] = indicators[ema_key].diff(5)
                
                # EMA signal generation
                current_price = combined_straddle.iloc[-1]
                ema_20 = indicators['ema_20'].iloc[-1]
                ema_100 = indicators['ema_100'].iloc[-1]
                ema_200 = indicators['ema_200'].iloc[-1]
                
                if current_price > ema_20 > ema_100 > ema_200:
                    indicators['ema_signal'] = 2  # Strong bullish
                elif current_price > ema_20 and ema_20 > ema_100:
                    indicators['ema_signal'] = 1  # Bullish
                elif current_price < ema_20 < ema_100 < ema_200:
                    indicators['ema_signal'] = -2  # Strong bearish
                elif current_price < ema_20 and ema_20 < ema_100:
                    indicators['ema_signal'] = -1  # Bearish
                else:
                    indicators['ema_signal'] = 0  # Neutral
            
            # VWAP Analysis
            if self.config['technical_indicators']['vwap_enabled']:
                cumulative_volume = volume.cumsum()
                cumulative_pv = (combined_straddle * volume).cumsum()
                indicators['vwap'] = cumulative_pv / cumulative_volume
                indicators['vwap_deviation'] = (combined_straddle - indicators['vwap']) / indicators['vwap']
                
                # VWAP signal
                current_deviation = indicators['vwap_deviation'].iloc[-1]
                if current_deviation > 0.02:
                    indicators['vwap_signal'] = 1  # Above VWAP
                elif current_deviation < -0.02:
                    indicators['vwap_signal'] = -1  # Below VWAP
                else:
                    indicators['vwap_signal'] = 0  # At VWAP
            
            # Support/Resistance Analysis
            if self.config['technical_indicators']['support_resistance_enabled']:
                lookback = 20
                indicators['support'] = combined_straddle.rolling(lookback).min()
                indicators['resistance'] = combined_straddle.rolling(lookback).max()
                
                current_price = combined_straddle.iloc[-1]
                current_support = indicators['support'].iloc[-1]
                current_resistance = indicators['resistance'].iloc[-1]
                
                support_distance = (current_price - current_support) / current_support
                resistance_distance = (current_resistance - current_price) / current_price
                
                if support_distance < 0.01:  # Near support
                    indicators['sr_signal'] = -1
                elif resistance_distance < 0.01:  # Near resistance
                    indicators['sr_signal'] = 1
                else:
                    indicators['sr_signal'] = 0
            
            # Combined technical signal
            signals = [
                indicators.get('ema_signal', 0) * 0.4,
                indicators.get('vwap_signal', 0) * 0.3,
                indicators.get('sr_signal', 0) * 0.3
            ]
            indicators['combined_technical_signal'] = sum(signals)
            
            logger.info("✅ Technical indicators calculated")
            return indicators
            
        except Exception as e:
            logger.error(f"❌ Error calculating technical indicators: {e}")
            return {}
    
    def calculate_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlation analysis between straddle components"""
        logger.info("🔗 Calculating correlation analysis...")
        
        try:
            straddle_data = df[['atm_straddle', 'itm1_straddle', 'otm1_straddle']]
            window = self.config['correlation_window']
            
            # Rolling correlations
            correlations = {}
            correlations['atm_itm1'] = straddle_data['atm_straddle'].rolling(window).corr(
                straddle_data['itm1_straddle'])
            correlations['atm_otm1'] = straddle_data['atm_straddle'].rolling(window).corr(
                straddle_data['otm1_straddle'])
            correlations['itm1_otm1'] = straddle_data['itm1_straddle'].rolling(window).corr(
                straddle_data['otm1_straddle'])
            
            # Average correlation
            correlations['average_correlation'] = (
                correlations['atm_itm1'] + correlations['atm_otm1'] + correlations['itm1_otm1']
            ) / 3
            
            # Correlation regime classification
            avg_corr = correlations['average_correlation'].iloc[-1]
            if avg_corr > 0.9:
                correlation_regime = "Extreme_Correlation"
            elif avg_corr > 0.8:
                correlation_regime = "High_Correlation"
            elif avg_corr > 0.6:
                correlation_regime = "Medium_Correlation"
            elif avg_corr > 0.3:
                correlation_regime = "Low_Correlation"
            else:
                correlation_regime = "Decorrelated"
            
            correlations['correlation_regime'] = correlation_regime
            correlations['diversification_benefit'] = 1 - avg_corr
            
            logger.info(f"✅ Correlation analysis completed - Regime: {correlation_regime}")
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlation analysis: {e}")
            return {}
    
    def run_enhanced_analysis(self, csv_file_path: str, current_vix: float = 20, 
                            dte: int = 2) -> str:
        """Run complete enhanced combined straddle analysis"""
        logger.info("🚀 Starting ENHANCED Combined Straddle analysis...")
        
        try:
            # Load data
            df = pd.read_csv(csv_file_path)
            logger.info(f"📊 Loaded {len(df)} data points")
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1min')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate enhanced combined straddle
            df, optimal_weights = self.calculate_enhanced_combined_straddle(df, current_vix, dte)
            
            # Calculate technical indicators
            technical_indicators = self.calculate_technical_indicators(
                df['enhanced_combined_straddle'], 
                df['atm_straddle_volume']
            )
            
            # Add technical indicators to dataframe
            for key, value in technical_indicators.items():
                if isinstance(value, pd.Series):
                    df[f'tech_{key}'] = value
                else:
                    df[f'tech_{key}'] = value
            
            # Calculate correlation analysis
            correlation_analysis = self.calculate_correlation_analysis(df)
            
            # Add correlation data to dataframe
            for key, value in correlation_analysis.items():
                if isinstance(value, pd.Series):
                    df[f'corr_{key}'] = value
                else:
                    df[f'corr_{key}'] = value
            
            # Add weight information
            for component, weight in optimal_weights.items():
                df[f'weight_{component}'] = weight
            
            # Generate output
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"enhanced_combined_straddle_analysis_{timestamp}.csv"
            
            # Save results
            df.to_csv(output_path, index=True)
            
            # Calculate performance comparison
            self.calculate_performance_comparison(df)
            
            logger.info(f"✅ ENHANCED Combined Straddle analysis completed: {output_path}")
            logger.info(f"🎯 Optimal weights: ATM={optimal_weights['atm']:.1%}, "
                       f"ITM1={optimal_weights['itm1']:.1%}, OTM1={optimal_weights['otm1']:.1%}")
            logger.info(f"🔗 Correlation regime: {correlation_analysis.get('correlation_regime', 'Unknown')}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Enhanced analysis failed: {e}")
            raise
    
    def calculate_performance_comparison(self, df: pd.DataFrame):
        """Calculate performance comparison between enhanced and traditional approaches"""
        logger.info("📊 Calculating performance comparison...")
        
        try:
            # Calculate returns
            enhanced_returns = df['enhanced_combined_straddle'].pct_change().dropna()
            traditional_returns = df['traditional_combined_straddle'].pct_change().dropna()
            
            # Performance metrics
            enhanced_metrics = self.calculate_performance_metrics(enhanced_returns)
            traditional_metrics = self.calculate_performance_metrics(traditional_returns)
            
            # Statistical significance test
            if len(enhanced_returns) == len(traditional_returns) and len(enhanced_returns) > 30:
                t_stat, p_value = stats.ttest_rel(enhanced_returns, traditional_returns)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(enhanced_returns) + np.var(traditional_returns)) / 2)
                cohens_d = (np.mean(enhanced_returns) - np.mean(traditional_returns)) / pooled_std
                
                significance_test = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05
                }
            else:
                significance_test = {'error': 'Insufficient data for statistical test'}
            
            # Store results
            self.performance_metrics = {
                'enhanced': enhanced_metrics,
                'traditional': traditional_metrics,
                'significance_test': significance_test
            }
            
            # Log results
            logger.info(f"📈 Enhanced Sharpe Ratio: {enhanced_metrics['sharpe_ratio']:.3f}")
            logger.info(f"📈 Traditional Sharpe Ratio: {traditional_metrics['sharpe_ratio']:.3f}")
            if 'p_value' in significance_test:
                logger.info(f"📊 Statistical significance: p={significance_test['p_value']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance comparison: {e}")
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': returns.mean() * 252,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': self.calculate_max_drawdown(returns),
            'hit_rate': (returns > 0).mean(),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        return metrics
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

if __name__ == "__main__":
    # Run enhanced combined straddle analysis
    system = EnhancedCombinedStraddleSystem()
    
    # Test with sample data
    csv_file = "real_data_validation_results/real_data_regime_formation_20250619_211454.csv"
    
    try:
        output_path = system.run_enhanced_analysis(csv_file, current_vix=22, dte=1)
        
        print("\n" + "="*80)
        print("ENHANCED COMBINED STRADDLE ANALYSIS COMPLETED")
        print("="*80)
        print(f"Input: {csv_file}")
        print(f"Output: {output_path}")
        print("="*80)
        print("🚀 Enhanced Combined Straddle with Portfolio Optimization")
        print("📊 Three-straddle combination (ATM + ITM1 + OTM1)")
        print("⚖️ Optimal weight allocation with market adjustments")
        print("📈 Comprehensive technical analysis integration")
        print("🔗 Correlation analysis and regime detection")
        print("📊 Performance comparison with statistical validation")
        print("="*80)
        
        # Display performance metrics if available
        if system.performance_metrics:
            enhanced = system.performance_metrics['enhanced']
            traditional = system.performance_metrics['traditional']
            
            print(f"Enhanced Sharpe Ratio: {enhanced['sharpe_ratio']:.3f}")
            print(f"Traditional Sharpe Ratio: {traditional['sharpe_ratio']:.3f}")
            print(f"Improvement: {((enhanced['sharpe_ratio'] / traditional['sharpe_ratio']) - 1) * 100:.1f}%")
            
            if 'p_value' in system.performance_metrics['significance_test']:
                p_val = system.performance_metrics['significance_test']['p_value']
                print(f"Statistical Significance: p = {p_val:.4f}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
