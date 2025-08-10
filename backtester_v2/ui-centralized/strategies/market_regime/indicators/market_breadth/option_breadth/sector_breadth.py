"""
Sector Breadth - Sector-based Option Market Breadth Analysis
==========================================================

Analyzes market breadth across different sectors using option data.

Author: Market Regime Refactoring Team
Date: 2025-07-07
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set
import logging

logger = logging.getLogger(__name__)


class SectorBreadth:
    """Sector-based market breadth analyzer"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Sector Breadth analyzer"""
        self.sector_threshold = config.get('sector_threshold', 0.6)
        self.breadth_window = config.get('breadth_window', 20)
        self.min_sector_participation = config.get('min_sector_participation', 0.3)
        
        # Sector definitions (can be customized)
        self.sector_mapping = config.get('sector_mapping', {
            'NIFTY_BANK': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK'],
            'NIFTY_IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
            'NIFTY_PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'APOLLOHOSP'],
            'NIFTY_AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO'],
            'NIFTY_FMCG': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA'],
            'NIFTY_ENERGY': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID'],
            'NIFTY_METAL': ['TATASTEEL', 'HINDALCO', 'JSW', 'COALINDIA'],
            'NIFTY_REALTY': ['DLF', 'GODREJPROP', 'OBEROI', 'BRIGADE']
        })
        
        # Historical tracking
        self.sector_history = {
            'sector_breadth': [],
            'advancing_sectors': [],
            'declining_sectors': [],
            'timestamps': []
        }
        
        logger.info("SectorBreadth initialized")
    
    def analyze_sector_breadth(self, option_data: pd.DataFrame, underlying_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze sector breadth using option data
        
        Args:
            option_data: DataFrame with option data including symbol/sector info
            underlying_data: Optional underlying price data by sector
            
        Returns:
            Dict with sector breadth analysis
        """
        try:
            if option_data.empty:
                return self._get_default_sector_analysis()
            
            # Map symbols to sectors
            sector_data = self._map_symbols_to_sectors(option_data)
            
            # Calculate sector metrics
            sector_metrics = self._calculate_sector_metrics(sector_data)
            
            # Analyze sector participation
            sector_participation = self._analyze_sector_participation(sector_data)
            
            # Detect sector rotation
            sector_rotation = self._detect_sector_rotation(sector_data, underlying_data)
            
            # Calculate sector divergences
            sector_divergences = self._calculate_sector_divergences(sector_data)
            
            # Generate sector signals
            sector_signals = self._generate_sector_signals(sector_metrics, sector_participation, sector_rotation)
            
            # Update historical tracking
            self._update_sector_history(sector_metrics, sector_participation)
            
            return {
                'sector_metrics': sector_metrics,
                'sector_participation': sector_participation,
                'sector_rotation': sector_rotation,
                'sector_divergences': sector_divergences,
                'sector_signals': sector_signals,
                'breadth_score': self._calculate_sector_breadth_score(sector_metrics, sector_participation)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sector breadth: {e}")
            return self._get_default_sector_analysis()
    
    def _map_symbols_to_sectors(self, option_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Map option symbols to their respective sectors"""
        try:
            sector_data = {}
            
            # Initialize sector dataframes
            for sector in self.sector_mapping.keys():
                sector_data[sector] = pd.DataFrame()
            
            # Map each row to appropriate sector
            if 'symbol' in option_data.columns:
                for sector, symbols in self.sector_mapping.items():
                    sector_options = option_data[option_data['symbol'].isin(symbols)]
                    if not sector_options.empty:
                        sector_data[sector] = sector_options
            elif 'underlying' in option_data.columns:
                for sector, symbols in self.sector_mapping.items():
                    sector_options = option_data[option_data['underlying'].isin(symbols)]
                    if not sector_options.empty:
                        sector_data[sector] = sector_options
            else:
                # Fallback: try to infer from available columns
                logger.warning(\"No symbol/underlying column found, using fallback sector mapping\")
                # Assume single sector for all data
                sector_data['MIXED'] = option_data
            
            return sector_data
            
        except Exception as e:
            logger.error(f\"Error mapping symbols to sectors: {e}\")
            return {'MIXED': option_data}
    
    def _calculate_sector_metrics(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        \"\"\"Calculate comprehensive metrics for each sector\"\"\"
        try:
            metrics = {}
            
            for sector, data in sector_data.items():
                if data.empty:
                    continue
                
                sector_metrics = {
                    'total_volume': float(data['volume'].sum()),
                    'total_oi': float(data['oi'].sum()) if 'oi' in data.columns else 0.0,
                    'avg_iv': float(data['iv'].mean()) if 'iv' in data.columns else 0.0,
                    'option_count': len(data),
                    'unique_strikes': len(data['strike'].unique()) if 'strike' in data.columns else 0,
                    'call_put_ratio': 0.0
                }
                
                # Call/Put analysis
                if 'option_type' in data.columns:
                    calls = data[data['option_type'] == 'CE']
                    puts = data[data['option_type'] == 'PE']
                    
                    call_volume = calls['volume'].sum()
                    put_volume = puts['volume'].sum()
                    
                    sector_metrics['call_volume'] = float(call_volume)
                    sector_metrics['put_volume'] = float(put_volume)
                    sector_metrics['call_put_ratio'] = float(call_volume / put_volume) if put_volume > 0 else 0.0
                
                # Moneyness distribution
                if 'moneyness' in data.columns:
                    itm_volume = data[data['moneyness'] < 1.0]['volume'].sum()
                    otm_volume = data[data['moneyness'] > 1.0]['volume'].sum()
                    
                    sector_metrics['itm_volume'] = float(itm_volume)
                    sector_metrics['otm_volume'] = float(otm_volume)
                    sector_metrics['itm_otm_ratio'] = float(itm_volume / otm_volume) if otm_volume > 0 else 0.0
                
                # Activity level classification
                total_volume = sector_metrics['total_volume']
                if total_volume > 0:
                    # Calculate percentiles across all sectors for relative activity
                    all_volumes = [metrics[s]['total_volume'] for s in metrics.keys() if 'total_volume' in metrics[s]]
                    if all_volumes:
                        avg_volume = np.mean(all_volumes)
                        if total_volume > avg_volume * 1.5:
                            sector_metrics['activity_level'] = 'high'
                        elif total_volume < avg_volume * 0.5:
                            sector_metrics['activity_level'] = 'low'
                        else:
                            sector_metrics['activity_level'] = 'normal'
                    else:
                        sector_metrics['activity_level'] = 'normal'
                
                metrics[sector] = sector_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f\"Error calculating sector metrics: {e}\")
            return {}
    
    def _analyze_sector_participation(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        \"\"\"Analyze participation patterns across sectors\"\"\"
        try:
            participation = {
                'active_sectors': [],
                'participating_sectors': 0,
                'total_sectors': 0,
                'participation_ratio': 0.0,
                'sector_concentration': {}
            }
            
            total_volume = 0
            sector_volumes = {}
            
            # Calculate volume by sector
            for sector, data in sector_data.items():
                if not data.empty:
                    volume = data['volume'].sum()
                    sector_volumes[sector] = volume
                    total_volume += volume
                    
                    if volume > 0:
                        participation['active_sectors'].append(sector)
            
            participation['participating_sectors'] = len(participation['active_sectors'])
            participation['total_sectors'] = len([s for s in sector_data.keys() if not sector_data[s].empty])
            
            if participation['total_sectors'] > 0:
                participation['participation_ratio'] = float(participation['participating_sectors'] / participation['total_sectors'])
            
            # Calculate sector concentration
            if total_volume > 0:
                for sector, volume in sector_volumes.items():
                    participation['sector_concentration'][sector] = float(volume / total_volume)
                
                # Top 3 sector concentration
                top_3_volume = sum(sorted(sector_volumes.values(), reverse=True)[:3])
                participation['top_3_concentration'] = float(top_3_volume / total_volume)
            
            # Sector breadth classification
            if participation['participation_ratio'] >= 0.8:
                participation['breadth_classification'] = 'broad'
            elif participation['participation_ratio'] >= 0.5:
                participation['breadth_classification'] = 'moderate'
            else:
                participation['breadth_classification'] = 'narrow'
            
            return participation
            
        except Exception as e:
            logger.error(f\"Error analyzing sector participation: {e}\")
            return {'participating_sectors': 0, 'participation_ratio': 0.0}
    
    def _detect_sector_rotation(self, sector_data: Dict[str, pd.DataFrame], underlying_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        \"\"\"Detect sector rotation patterns\"\"\"
        try:
            rotation = {
                'rotation_signals': [],
                'momentum_shifts': {},
                'rotation_strength': 0.0
            }
            
            # Compare current volume distribution with historical
            current_volumes = {}
            for sector, data in sector_data.items():
                if not data.empty:
                    current_volumes[sector] = data['volume'].sum()
            
            if self.sector_history['sector_breadth'] and current_volumes:
                # Get historical average volumes
                historical_avg = {}
                for entry in self.sector_history['sector_breadth'][-self.breadth_window:]:
                    for sector, volume in entry.items():
                        if sector not in historical_avg:
                            historical_avg[sector] = []
                        historical_avg[sector].append(volume)
                
                # Calculate momentum shifts
                for sector in current_volumes.keys():
                    if sector in historical_avg and historical_avg[sector]:
                        current_vol = current_volumes[sector]
                        avg_vol = np.mean(historical_avg[sector])
                        
                        if avg_vol > 0:
                            momentum_shift = (current_vol - avg_vol) / avg_vol
                            rotation['momentum_shifts'][sector] = float(momentum_shift)
                            
                            # Detect significant shifts
                            if momentum_shift > 0.5:
                                rotation['rotation_signals'].append(f'{sector}_inflow')
                            elif momentum_shift < -0.5:
                                rotation['rotation_signals'].append(f'{sector}_outflow')
                
                # Calculate overall rotation strength
                if rotation['momentum_shifts']:
                    momentum_values = list(rotation['momentum_shifts'].values())
                    rotation['rotation_strength'] = float(np.std(momentum_values))
            
            # Underlying price-based rotation (if available)
            if underlying_data is not None:
                rotation['price_rotation'] = self._analyze_price_rotation(underlying_data)
            
            return rotation
            
        except Exception as e:
            logger.error(f\"Error detecting sector rotation: {e}\")
            return {'rotation_signals': [], 'rotation_strength': 0.0}
    
    def _analyze_price_rotation(self, underlying_data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Analyze sector rotation based on underlying price movements\"\"\"
        try:
            price_rotation = {}
            
            if 'symbol' in underlying_data.columns and 'price_change' in underlying_data.columns:
                # Map symbols to sectors and calculate sector performance
                sector_performance = {}
                
                for sector, symbols in self.sector_mapping.items():
                    sector_data = underlying_data[underlying_data['symbol'].isin(symbols)]
                    if not sector_data.empty:
                        avg_performance = sector_data['price_change'].mean()
                        sector_performance[sector] = float(avg_performance)
                
                if sector_performance:
                    # Identify leaders and laggards
                    sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
                    
                    price_rotation['sector_leaders'] = sorted_sectors[:3]
                    price_rotation['sector_laggards'] = sorted_sectors[-3:]
                    price_rotation['performance_spread'] = float(sorted_sectors[0][1] - sorted_sectors[-1][1])
            
            return price_rotation
            
        except Exception as e:
            logger.error(f\"Error analyzing price rotation: {e}\")
            return {}
    
    def _calculate_sector_divergences(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        \"\"\"Calculate divergences between sectors\"\"\"
        try:
            divergences = {
                'divergence_signals': [],
                'divergence_count': 0,
                'max_divergence': 0.0
            }
            
            # Calculate IV divergences between sectors
            sector_ivs = {}
            for sector, data in sector_data.items():
                if not data.empty and 'iv' in data.columns:
                    sector_ivs[sector] = data['iv'].mean()
            
            if len(sector_ivs) >= 2:
                iv_values = list(sector_ivs.values())
                iv_spread = max(iv_values) - min(iv_values)
                iv_std = np.std(iv_values)
                
                divergences['iv_spread'] = float(iv_spread)
                divergences['iv_divergence'] = float(iv_std)
                divergences['max_divergence'] = max(divergences['max_divergence'], iv_std)
                
                # Detect extreme IV divergences
                if iv_std > 0.05:  # 5% IV divergence threshold
                    divergences['divergence_signals'].append('extreme_iv_divergence')
            
            # Calculate volume concentration divergences
            sector_volumes = {}
            total_volume = 0
            
            for sector, data in sector_data.items():
                if not data.empty:
                    volume = data['volume'].sum()
                    sector_volumes[sector] = volume
                    total_volume += volume
            
            if total_volume > 0 and len(sector_volumes) >= 2:
                # Calculate concentration ratio
                volume_shares = [vol / total_volume for vol in sector_volumes.values()]
                concentration = sum(sorted(volume_shares, reverse=True)[:2])  # Top 2 sectors
                
                divergences['volume_concentration'] = float(concentration)
                
                if concentration > 0.8:  # 80% concentration in top 2 sectors
                    divergences['divergence_signals'].append('extreme_volume_concentration')
                    divergences['max_divergence'] = max(divergences['max_divergence'], concentration)
            
            divergences['divergence_count'] = len(divergences['divergence_signals'])
            
            return divergences
            
        except Exception as e:
            logger.error(f\"Error calculating sector divergences: {e}\")
            return {'divergence_signals': [], 'divergence_count': 0}
    
    def _generate_sector_signals(self, sector_metrics: Dict[str, Any], participation: Dict[str, Any], rotation: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Generate actionable sector breadth signals\"\"\"
        try:
            signals = {
                'primary_signal': 'neutral',
                'signal_strength': 0.0,
                'sector_signals': [],
                'breadth_implications': []
            }
            
            # Participation signals
            participation_ratio = participation.get('participation_ratio', 0.0)
            if participation_ratio >= 0.8:
                signals['sector_signals'].append('broad_sector_participation')
                signals['breadth_implications'].append('wide_market_engagement')
                signals['signal_strength'] += 0.3
            elif participation_ratio <= 0.3:
                signals['sector_signals'].append('narrow_sector_participation')
                signals['breadth_implications'].append('concentrated_market_activity')
                signals['signal_strength'] += 0.3
            
            # Concentration signals
            concentration = participation.get('top_3_concentration', 0.0)
            if concentration > 0.8:
                signals['sector_signals'].append('high_sector_concentration')
                signals['breadth_implications'].append('narrow_leadership')
                signals['signal_strength'] += 0.2
            elif concentration < 0.5:
                signals['sector_signals'].append('distributed_sector_activity')
                signals['breadth_implications'].append('broad_leadership')
                signals['signal_strength'] += 0.2
            
            # Rotation signals
            rotation_strength = rotation.get('rotation_strength', 0.0)
            if rotation_strength > 0.5:
                signals['sector_signals'].append('active_sector_rotation')
                signals['breadth_implications'].append('shifting_market_leadership')
                signals['signal_strength'] += 0.3
            
            rotation_signals = rotation.get('rotation_signals', [])
            if len(rotation_signals) >= 3:
                signals['sector_signals'].append('multiple_sector_shifts')
                signals['breadth_implications'].append('broad_rotation_pattern')
                signals['signal_strength'] += 0.2
            
            # Determine primary signal
            if signals['signal_strength'] > 0.6:
                if any('broad' in sig for sig in signals['sector_signals']):
                    signals['primary_signal'] = 'expanding_sector_breadth'
                elif any('narrow' in sig for sig in signals['sector_signals']):
                    signals['primary_signal'] = 'contracting_sector_breadth'
                elif any('rotation' in sig for sig in signals['sector_signals']):
                    signals['primary_signal'] = 'sector_rotation_active'
                else:
                    signals['primary_signal'] = 'sector_breadth_shift'
            elif signals['signal_strength'] > 0.3:
                signals['primary_signal'] = 'moderate_sector_change'
            
            return signals
            
        except Exception as e:
            logger.error(f\"Error generating sector signals: {e}\")
            return {'primary_signal': 'neutral', 'signal_strength': 0.0}
    
    def _calculate_sector_breadth_score(self, sector_metrics: Dict[str, Any], participation: Dict[str, Any]) -> float:
        \"\"\"Calculate overall sector breadth score\"\"\"
        try:
            score = 0.5  # Base neutral score
            
            # Participation ratio contribution (40%)
            participation_ratio = participation.get('participation_ratio', 0.0)
            score += (participation_ratio - 0.5) * 0.4
            
            # Volume distribution contribution (30%)
            concentration = participation.get('top_3_concentration', 0.5)
            distribution_score = max(1.0 - concentration, 0.0)  # Lower concentration = higher breadth
            score += (distribution_score - 0.5) * 0.3
            
            # Activity balance contribution (20%)
            if sector_metrics:
                activity_levels = [metrics.get('activity_level', 'normal') for metrics in sector_metrics.values()]
                high_activity = sum(1 for level in activity_levels if level == 'high')
                total_active = len(activity_levels)
                
                if total_active > 0:
                    activity_balance = 1.0 - abs(high_activity / total_active - 0.5) * 2
                    score += (activity_balance - 0.5) * 0.2
            
            # Sector count contribution (10%)
            active_sectors = participation.get('participating_sectors', 0)
            total_sectors = participation.get('total_sectors', 8)  # Default sector count
            
            if total_sectors > 0:
                sector_score = active_sectors / total_sectors
                score += (sector_score - 0.5) * 0.1
            
            return max(min(float(score), 1.0), 0.0)
            
        except Exception as e:
            logger.error(f\"Error calculating sector breadth score: {e}\")
            return 0.5
    
    def _update_sector_history(self, sector_metrics: Dict[str, Any], participation: Dict[str, Any]):
        \"\"\"Update historical sector tracking\"\"\"
        try:
            # Update sector volume history
            current_volumes = {}
            for sector, metrics in sector_metrics.items():
                current_volumes[sector] = metrics.get('total_volume', 0.0)
            
            self.sector_history['sector_breadth'].append(current_volumes)
            if len(self.sector_history['sector_breadth']) > self.breadth_window * 2:
                self.sector_history['sector_breadth'].pop(0)
            
            # Update advancing/declining sectors
            active_sectors = participation.get('active_sectors', [])
            self.sector_history['advancing_sectors'].append(len(active_sectors))
            if len(self.sector_history['advancing_sectors']) > self.breadth_window:
                self.sector_history['advancing_sectors'].pop(0)
            
        except Exception as e:
            logger.error(f\"Error updating sector history: {e}\")
    
    def _get_default_sector_analysis(self) -> Dict[str, Any]:
        \"\"\"Get default sector analysis when data is insufficient\"\"\"
        return {
            'sector_metrics': {},
            'sector_participation': {'participating_sectors': 0, 'participation_ratio': 0.0},
            'sector_rotation': {'rotation_signals': [], 'rotation_strength': 0.0},
            'sector_divergences': {'divergence_signals': [], 'divergence_count': 0},
            'sector_signals': {'primary_signal': 'neutral', 'signal_strength': 0.0},
            'breadth_score': 0.5
        }
    
    def get_sector_summary(self) -> Dict[str, Any]:
        \"\"\"Get summary of sector breadth analysis\"\"\"
        try:
            return {
                'sector_mapping': self.sector_mapping,
                'history_length': len(self.sector_history['sector_breadth']),
                'average_participation': np.mean(self.sector_history['advancing_sectors']) if self.sector_history['advancing_sectors'] else 0.0,
                'total_sectors_tracked': len(self.sector_mapping),
                'analysis_config': {
                    'sector_threshold': self.sector_threshold,
                    'breadth_window': self.breadth_window,
                    'min_sector_participation': self.min_sector_participation
                }
            }
            
        except Exception as e:
            logger.error(f\"Error getting sector summary: {e}\")
            return {'status': 'error', 'error': str(e)}