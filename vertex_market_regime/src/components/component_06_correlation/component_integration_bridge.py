"""
Component Integration Bridge for Component 6

Provides seamless integration with existing Components 1-5 outputs,
extracting and transforming data for correlation analysis and feature engineering.

Features:
- Component 1-5 output parsing and extraction
- Data alignment and synchronization
- Feature mapping and transformation
- Performance-optimized data flows
- Error handling and fallback mechanisms

🎯 PURE DATA INTEGRATION - NO BUSINESS LOGIC
Focuses on data transformation and alignment only.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import time
import warnings
import asyncio

warnings.filterwarnings('ignore')


@dataclass
class ComponentDataExtract:
    """Extracted data from a single component"""
    component_id: int
    component_name: str
    primary_time_series: pd.DataFrame
    secondary_metrics: Dict[str, Any]
    feature_vector: np.ndarray
    confidence_score: float
    processing_time_ms: float
    timestamp: datetime


@dataclass
class IntegratedComponentData:
    """Integrated data from all components"""
    components_data: Dict[int, ComponentDataExtract]
    aligned_time_series: pd.DataFrame
    common_time_index: pd.DatetimeIndex
    cross_component_features: Dict[str, np.ndarray]
    integration_quality_score: float
    total_processing_time_ms: float
    timestamp: datetime


class ComponentDataExtractor:
    """Extracts and standardizes data from component analysis results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Data extraction settings
        self.min_data_points = config.get('min_data_points', 20)
        self.time_alignment_tolerance = config.get('time_alignment_tolerance', '1min')
        self.feature_standardization = config.get('feature_standardization', True)
        
        # Component-specific extraction mappings
        self.component_extractors = {
            1: self._extract_component_1_data,
            2: self._extract_component_2_data,
            3: self._extract_component_3_data,
            4: self._extract_component_4_data,
            5: self._extract_component_5_data
        }
        
        self.logger.info("Component data extractor initialized")

    def extract_component_data(self, component_id: int, component_result: Any) -> ComponentDataExtract:
        """
        Extract standardized data from component analysis result
        
        Args:
            component_id: ID of the component (1-5)
            component_result: Component analysis result object
            
        Returns:
            ComponentDataExtract with standardized data
        """
        start_time = time.time()
        
        try:
            if component_id in self.component_extractors:
                extractor = self.component_extractors[component_id]
                component_data = extractor(component_result)
            else:
                component_data = self._extract_generic_component_data(component_id, component_result)
            
            processing_time = (time.time() - start_time) * 1000
            
            return ComponentDataExtract(
                component_id=component_id,
                component_name=component_data.get('name', f'Component_{component_id}'),
                primary_time_series=component_data.get('time_series', pd.DataFrame()),
                secondary_metrics=component_data.get('metrics', {}),
                feature_vector=component_data.get('features', np.array([])),
                confidence_score=component_data.get('confidence', 0.5),
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Error extracting Component {component_id} data: {e}")
            
            return ComponentDataExtract(
                component_id=component_id,
                component_name=f'Component_{component_id}',
                primary_time_series=pd.DataFrame(),
                secondary_metrics={},
                feature_vector=np.array([]),
                confidence_score=0.0,
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )

    def _extract_component_1_data(self, result: Any) -> Dict[str, Any]:
        """Extract Component 1 (Triple Straddle) data"""
        
        try:
            extracted_data = {
                'name': 'Triple_Straddle_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.5
            }
            
            # Extract straddle time series
            if hasattr(result, 'straddle_time_series'):
                straddle_ts = result.straddle_time_series
                
                # Build time series DataFrame
                time_series_data = {}
                if hasattr(straddle_ts, 'atm_straddle_prices') and straddle_ts.atm_straddle_prices is not None:
                    time_series_data['atm_premium'] = straddle_ts.atm_straddle_prices
                if hasattr(straddle_ts, 'itm1_straddle_prices') and straddle_ts.itm1_straddle_prices is not None:
                    time_series_data['itm1_premium'] = straddle_ts.itm1_straddle_prices
                if hasattr(straddle_ts, 'otm1_straddle_prices') and straddle_ts.otm1_straddle_prices is not None:
                    time_series_data['otm1_premium'] = straddle_ts.otm1_straddle_prices
                if hasattr(straddle_ts, 'timestamps') and straddle_ts.timestamps is not None:
                    time_series_data['timestamp'] = straddle_ts.timestamps
                
                if time_series_data:
                    extracted_data['time_series'] = pd.DataFrame(time_series_data)
                    if 'timestamp' in time_series_data:
                        extracted_data['time_series'].set_index('timestamp', inplace=True)
            
            # Extract secondary metrics
            metrics = {}
            if hasattr(result, 'weighting_analysis'):
                weighting = result.weighting_analysis
                if hasattr(weighting, 'component_weights'):
                    metrics['component_weights'] = weighting.component_weights
                if hasattr(weighting, 'confidence_score'):
                    extracted_data['confidence'] = weighting.confidence_score
            
            if hasattr(result, 'ema_analysis'):
                ema = result.ema_analysis
                if hasattr(ema, 'trend_strength'):
                    metrics['ema_trend_strength'] = ema.trend_strength
                if hasattr(ema, 'ema_values'):
                    metrics['ema_values'] = ema.ema_values
            
            if hasattr(result, 'vwap_analysis'):
                vwap = result.vwap_analysis
                if hasattr(vwap, 'vwap_values'):
                    metrics['vwap_values'] = vwap.vwap_values
                if hasattr(vwap, 'volume_profile'):
                    metrics['volume_profile'] = vwap.volume_profile
            
            extracted_data['metrics'] = metrics
            
            # Extract features
            if hasattr(result, 'features') and hasattr(result.features, 'features'):
                extracted_data['features'] = result.features.features
            elif hasattr(result, 'feature_vector'):
                extracted_data['features'] = result.feature_vector
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error extracting Component 1 data: {e}")
            return {
                'name': 'Triple_Straddle_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.0
            }

    def _extract_component_2_data(self, result: Any) -> Dict[str, Any]:
        """Extract Component 2 (Greeks Sentiment) data"""
        
        try:
            extracted_data = {
                'name': 'Greeks_Sentiment_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.5
            }
            
            # Extract Greeks data
            greeks_data = {}
            if hasattr(result, 'greeks_analysis'):
                greeks = result.greeks_analysis
                
                # Extract Greek values
                for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
                    if hasattr(greeks, f'{greek}_values'):
                        greeks_data[greek] = getattr(greeks, f'{greek}_values')
                    elif hasattr(greeks, greek):
                        greeks_data[greek] = getattr(greeks, greek)
                
                # Extract timestamps if available
                if hasattr(greeks, 'timestamps'):
                    greeks_data['timestamp'] = greeks.timestamps
                
                if greeks_data:
                    extracted_data['time_series'] = pd.DataFrame(greeks_data)
                    if 'timestamp' in greeks_data:
                        extracted_data['time_series'].set_index('timestamp', inplace=True)
            
            # Extract sentiment metrics
            metrics = {}
            if hasattr(result, 'sentiment_analysis'):
                sentiment = result.sentiment_analysis
                if hasattr(sentiment, 'sentiment_score'):
                    metrics['sentiment_score'] = sentiment.sentiment_score
                if hasattr(sentiment, 'volume_weighted_sentiment'):
                    metrics['volume_weighted_sentiment'] = sentiment.volume_weighted_sentiment
                if hasattr(sentiment, 'confidence_score'):
                    extracted_data['confidence'] = sentiment.confidence_score
            
            # Extract volume weighting data
            if hasattr(result, 'volume_weighting'):
                volume = result.volume_weighting
                if hasattr(volume, 'weighted_greeks'):
                    metrics['weighted_greeks'] = volume.weighted_greeks
                if hasattr(volume, 'volume_profile'):
                    metrics['volume_profile'] = volume.volume_profile
            
            extracted_data['metrics'] = metrics
            
            # Extract features
            if hasattr(result, 'features') and hasattr(result.features, 'features'):
                extracted_data['features'] = result.features.features
            elif hasattr(result, 'feature_vector'):
                extracted_data['features'] = result.feature_vector
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error extracting Component 2 data: {e}")
            return {
                'name': 'Greeks_Sentiment_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.0
            }

    def _extract_component_3_data(self, result: Any) -> Dict[str, Any]:
        """Extract Component 3 (OI/PA Trending) data"""
        
        try:
            extracted_data = {
                'name': 'OI_PA_Trending_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.5
            }
            
            # Extract OI/PA time series
            oi_data = {}
            if hasattr(result, 'oi_analysis'):
                oi = result.oi_analysis
                
                # Extract OI values
                for oi_type in ['ce_oi', 'pe_oi', 'total_oi', 'ce_volume', 'pe_volume', 'total_volume']:
                    if hasattr(oi, oi_type):
                        oi_data[oi_type] = getattr(oi, oi_type)
                
                # Extract timestamps
                if hasattr(oi, 'timestamps'):
                    oi_data['timestamp'] = oi.timestamps
                
                if oi_data:
                    extracted_data['time_series'] = pd.DataFrame(oi_data)
                    if 'timestamp' in oi_data:
                        extracted_data['time_series'].set_index('timestamp', inplace=True)
            
            # Extract institutional flow metrics
            metrics = {}
            if hasattr(result, 'institutional_flow'):
                flow = result.institutional_flow
                if hasattr(flow, 'flow_direction'):
                    metrics['flow_direction'] = flow.flow_direction
                if hasattr(flow, 'flow_strength'):
                    metrics['flow_strength'] = flow.flow_strength
                if hasattr(flow, 'confidence_score'):
                    extracted_data['confidence'] = flow.confidence_score
            
            # Extract divergence analysis
            if hasattr(result, 'divergence_analysis'):
                divergence = result.divergence_analysis
                if hasattr(divergence, 'price_oi_divergence'):
                    metrics['price_oi_divergence'] = divergence.price_oi_divergence
                if hasattr(divergence, 'volume_oi_divergence'):
                    metrics['volume_oi_divergence'] = divergence.volume_oi_divergence
            
            extracted_data['metrics'] = metrics
            
            # Extract features
            if hasattr(result, 'features') and hasattr(result.features, 'features'):
                extracted_data['features'] = result.features.features
            elif hasattr(result, 'feature_vector'):
                extracted_data['features'] = result.feature_vector
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error extracting Component 3 data: {e}")
            return {
                'name': 'OI_PA_Trending_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.0
            }

    def _extract_component_4_data(self, result: Any) -> Dict[str, Any]:
        """Extract Component 4 (IV Skew) data"""
        
        try:
            extracted_data = {
                'name': 'IV_Skew_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.5
            }
            
            # Extract IV skew time series
            iv_data = {}
            if hasattr(result, 'iv_skew_result'):
                iv_skew = result.iv_skew_result
                
                # Extract IV values
                for iv_type in ['atm_iv', 'itm_iv', 'otm_iv', 'skew_value', 'smile_curvature']:
                    if hasattr(iv_skew, iv_type):
                        iv_data[iv_type] = getattr(iv_skew, iv_type)
                
                # Extract DTE information
                if hasattr(iv_skew, 'dte_values'):
                    iv_data['dte'] = iv_skew.dte_values
                
                # Extract timestamps
                if hasattr(iv_skew, 'timestamps'):
                    iv_data['timestamp'] = iv_skew.timestamps
                
                if iv_data:
                    extracted_data['time_series'] = pd.DataFrame(iv_data)
                    if 'timestamp' in iv_data:
                        extracted_data['time_series'].set_index('timestamp', inplace=True)
            
            # Extract regime classification metrics
            metrics = {}
            if hasattr(result, 'regime_classification_result'):
                regime = result.regime_classification_result
                if hasattr(regime, 'regime_classification'):
                    metrics['regime_classification'] = regime.regime_classification
                if hasattr(regime, 'regime_confidence'):
                    metrics['regime_confidence'] = regime.regime_confidence
                    extracted_data['confidence'] = regime.regime_confidence
                if hasattr(regime, 'regime_probabilities'):
                    metrics['regime_probabilities'] = regime.regime_probabilities
            
            # Extract DTE framework metrics
            if hasattr(result, 'dte_framework_result'):
                dte_framework = result.dte_framework_result
                if hasattr(dte_framework, 'dte_specific_metrics'):
                    metrics['dte_specific_metrics'] = dte_framework.dte_specific_metrics
                if hasattr(dte_framework, 'term_structure'):
                    metrics['term_structure'] = dte_framework.term_structure
            
            extracted_data['metrics'] = metrics
            
            # Extract features
            if hasattr(result, 'feature_vector') and hasattr(result.feature_vector, 'features'):
                extracted_data['features'] = result.feature_vector.features
            elif hasattr(result, 'features'):
                extracted_data['features'] = result.features
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error extracting Component 4 data: {e}")
            return {
                'name': 'IV_Skew_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.0
            }

    def _extract_component_5_data(self, result: Any) -> Dict[str, Any]:
        """Extract Component 5 (ATR-EMA-CPR) data"""
        
        try:
            extracted_data = {
                'name': 'ATR_EMA_CPR_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.5
            }
            
            # Extract technical indicators time series
            technical_data = {}
            if hasattr(result, 'technical_analysis'):
                tech = result.technical_analysis
                
                # Extract technical indicators
                for indicator in ['atr', 'ema', 'cpr_pivot', 'cpr_bc', 'cpr_tc']:
                    if hasattr(tech, indicator):
                        technical_data[indicator] = getattr(tech, indicator)
                    elif hasattr(tech, f'{indicator}_values'):
                        technical_data[indicator] = getattr(tech, f'{indicator}_values')
                
                # Extract dual asset data (straddle vs underlying)
                if hasattr(tech, 'straddle_atr'):
                    technical_data['atr_straddle'] = tech.straddle_atr
                if hasattr(tech, 'underlying_atr'):
                    technical_data['atr_underlying'] = tech.underlying_atr
                
                # Extract timestamps
                if hasattr(tech, 'timestamps'):
                    technical_data['timestamp'] = tech.timestamps
                
                if technical_data:
                    extracted_data['time_series'] = pd.DataFrame(technical_data)
                    if 'timestamp' in technical_data:
                        extracted_data['time_series'].set_index('timestamp', inplace=True)
            
            # Extract regime classification metrics
            metrics = {}
            if hasattr(result, 'regime_classification'):
                regime = result.regime_classification
                if hasattr(regime, 'regime_class'):
                    metrics['regime_class'] = regime.regime_class
                if hasattr(regime, 'confidence_score'):
                    metrics['regime_confidence'] = regime.confidence_score
                    extracted_data['confidence'] = regime.confidence_score
                if hasattr(regime, 'volatility_regime'):
                    metrics['volatility_regime'] = regime.volatility_regime
            
            # Extract confluence metrics
            if hasattr(result, 'confluence_analysis'):
                confluence = result.confluence_analysis
                if hasattr(confluence, 'confluence_score'):
                    metrics['confluence_score'] = confluence.confluence_score
                if hasattr(confluence, 'support_resistance_levels'):
                    metrics['support_resistance_levels'] = confluence.support_resistance_levels
            
            extracted_data['metrics'] = metrics
            
            # Extract features
            if hasattr(result, 'features') and hasattr(result.features, 'features'):
                extracted_data['features'] = result.features.features
            elif hasattr(result, 'feature_vector'):
                extracted_data['features'] = result.feature_vector
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error extracting Component 5 data: {e}")
            return {
                'name': 'ATR_EMA_CPR_Analysis',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.0
            }

    def _extract_generic_component_data(self, component_id: int, result: Any) -> Dict[str, Any]:
        """Extract data from generic component result"""
        
        try:
            extracted_data = {
                'name': f'Component_{component_id}',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.5
            }
            
            # Try to extract common attributes
            if hasattr(result, 'confidence'):
                extracted_data['confidence'] = result.confidence
            elif hasattr(result, 'confidence_score'):
                extracted_data['confidence'] = result.confidence_score
            
            # Try to extract time series data
            if hasattr(result, 'time_series'):
                extracted_data['time_series'] = result.time_series
            elif hasattr(result, 'data'):
                if isinstance(result.data, pd.DataFrame):
                    extracted_data['time_series'] = result.data
            
            # Try to extract features
            if hasattr(result, 'features'):
                if hasattr(result.features, 'features'):
                    extracted_data['features'] = result.features.features
                else:
                    extracted_data['features'] = result.features
            elif hasattr(result, 'feature_vector'):
                extracted_data['features'] = result.feature_vector
            
            # Try to extract metrics
            if hasattr(result, 'metrics'):
                extracted_data['metrics'] = result.metrics
            elif hasattr(result, 'analysis_result'):
                extracted_data['metrics'] = result.analysis_result
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error extracting generic component {component_id} data: {e}")
            return {
                'name': f'Component_{component_id}',
                'time_series': pd.DataFrame(),
                'metrics': {},
                'features': np.array([]),
                'confidence': 0.0
            }


class ComponentDataAligner:
    """Aligns and synchronizes data from multiple components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Alignment settings
        self.time_tolerance = config.get('time_tolerance', '1min')
        self.missing_data_strategy = config.get('missing_data_strategy', 'interpolate')
        self.min_overlap_ratio = config.get('min_overlap_ratio', 0.5)
        
        self.logger.info("Component data aligner initialized")

    def align_component_data(self, components_data: Dict[int, ComponentDataExtract]) -> IntegratedComponentData:
        """
        Align and integrate data from multiple components
        
        Args:
            components_data: Dict of component data extracts
            
        Returns:
            IntegratedComponentData with aligned and synchronized data
        """
        start_time = time.time()
        
        try:
            # Find common time range across all components
            common_time_index = self._find_common_time_index(components_data)
            
            # Align time series data
            aligned_time_series = self._align_time_series(components_data, common_time_index)
            
            # Extract cross-component features
            cross_component_features = self._extract_cross_component_features(components_data, aligned_time_series)
            
            # Calculate integration quality score
            integration_quality = self._calculate_integration_quality(components_data, aligned_time_series)
            
            total_processing_time = (time.time() - start_time) * 1000
            
            return IntegratedComponentData(
                components_data=components_data,
                aligned_time_series=aligned_time_series,
                common_time_index=common_time_index,
                cross_component_features=cross_component_features,
                integration_quality_score=integration_quality,
                total_processing_time_ms=total_processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Error aligning component data: {e}")
            
            return IntegratedComponentData(
                components_data=components_data,
                aligned_time_series=pd.DataFrame(),
                common_time_index=pd.DatetimeIndex([]),
                cross_component_features={},
                integration_quality_score=0.0,
                total_processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )

    def _find_common_time_index(self, components_data: Dict[int, ComponentDataExtract]) -> pd.DatetimeIndex:
        """Find common time index across all components"""
        
        try:
            all_time_indices = []
            
            for component_id, data in components_data.items():
                time_series = data.primary_time_series
                if len(time_series) > 0:
                    if isinstance(time_series.index, pd.DatetimeIndex):
                        all_time_indices.append(time_series.index)
                    elif 'timestamp' in time_series.columns:
                        time_index = pd.to_datetime(time_series['timestamp'])
                        all_time_indices.append(time_index)
                    elif 'time' in time_series.columns:
                        time_index = pd.to_datetime(time_series['time'])
                        all_time_indices.append(time_index)
            
            if not all_time_indices:
                # Return empty DatetimeIndex if no time data found
                return pd.DatetimeIndex([])
            
            # Find intersection of all time indices
            common_index = all_time_indices[0]
            for time_index in all_time_indices[1:]:
                common_index = common_index.intersection(time_index)
            
            # If intersection is too small, use union with tolerance
            if len(common_index) < self.min_overlap_ratio * max(len(idx) for idx in all_time_indices):
                # Use union and resample to common frequency
                all_times = pd.DatetimeIndex([])
                for time_index in all_time_indices:
                    all_times = all_times.union(time_index)
                
                # Resample to 1-minute frequency (or config frequency)
                if len(all_times) > 0:
                    freq = self.config.get('alignment_frequency', '1min')
                    start_time = all_times.min()
                    end_time = all_times.max()
                    common_index = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            return common_index
            
        except Exception as e:
            self.logger.error(f"Error finding common time index: {e}")
            return pd.DatetimeIndex([])

    def _align_time_series(self, 
                         components_data: Dict[int, ComponentDataExtract],
                         common_time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Align time series data to common time index"""
        
        try:
            if len(common_time_index) == 0:
                return pd.DataFrame()
            
            aligned_data = {}
            
            for component_id, data in components_data.items():
                time_series = data.primary_time_series
                
                if len(time_series) == 0:
                    continue
                
                # Ensure time series has datetime index
                if not isinstance(time_series.index, pd.DatetimeIndex):
                    if 'timestamp' in time_series.columns:
                        time_series = time_series.set_index(pd.to_datetime(time_series['timestamp']))
                    elif 'time' in time_series.columns:
                        time_series = time_series.set_index(pd.to_datetime(time_series['time']))
                    else:
                        # Skip if no time information
                        continue
                
                # Resample/align to common index
                try:
                    aligned_series = time_series.reindex(common_time_index)
                    
                    # Handle missing data based on strategy
                    if self.missing_data_strategy == 'interpolate':
                        aligned_series = aligned_series.infer_objects(copy=False).interpolate(method='linear')
                    elif self.missing_data_strategy == 'forward_fill':
                        aligned_series = aligned_series.fillna(method='ffill')
                    elif self.missing_data_strategy == 'backward_fill':
                        aligned_series = aligned_series.fillna(method='bfill')
                    
                    # Add component prefix to column names
                    for col in aligned_series.columns:
                        aligned_data[f'comp_{component_id}_{col}'] = aligned_series[col]
                        
                except Exception as e:
                    self.logger.error(f"Error aligning Component {component_id} time series: {e}")
                    continue
            
            if aligned_data:
                aligned_df = pd.DataFrame(aligned_data, index=common_time_index)
                return aligned_df
            else:
                return pd.DataFrame(index=common_time_index)
                
        except Exception as e:
            self.logger.error(f"Error aligning time series: {e}")
            return pd.DataFrame()

    def _extract_cross_component_features(self, 
                                        components_data: Dict[int, ComponentDataExtract],
                                        aligned_time_series: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract cross-component correlation features"""
        
        cross_features = {}
        
        try:
            if len(aligned_time_series) < 20:  # Need minimum data for correlations
                return cross_features
            
            # Calculate pairwise correlations between component time series
            components = list(components_data.keys())
            
            for i in range(len(components)):
                for j in range(i + 1, len(components)):
                    comp_i, comp_j = components[i], components[j]
                    
                    # Find matching columns for each component
                    comp_i_cols = [col for col in aligned_time_series.columns if f'comp_{comp_i}_' in col]
                    comp_j_cols = [col for col in aligned_time_series.columns if f'comp_{comp_j}_' in col]
                    
                    if comp_i_cols and comp_j_cols:
                        # Calculate correlation between primary columns
                        col_i = comp_i_cols[0]  # Use first available column
                        col_j = comp_j_cols[0]
                        
                        try:
                            data_i = aligned_time_series[col_i].dropna()
                            data_j = aligned_time_series[col_j].dropna()
                            
                            # Find common indices
                            common_idx = data_i.index.intersection(data_j.index)
                            
                            if len(common_idx) >= 20:
                                corr_data_i = data_i[common_idx]
                                corr_data_j = data_j[common_idx]
                                
                                correlation = corr_data_i.corr(corr_data_j)
                                
                                if not np.isnan(correlation):
                                    feature_key = f'corr_comp_{comp_i}_vs_{comp_j}'
                                    cross_features[feature_key] = np.array([correlation], dtype=np.float32)
                                    
                        except Exception as e:
                            self.logger.error(f"Error calculating correlation between components {comp_i} and {comp_j}: {e}")
                            continue
            
            # Calculate additional cross-component features
            if len(aligned_time_series.columns) >= 2:
                # Overall correlation matrix statistics
                try:
                    corr_matrix = aligned_time_series.corr()
                    
                    # Extract upper triangle (excluding diagonal)
                    mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
                    upper_triangle = corr_matrix.where(mask).stack()
                    
                    if len(upper_triangle) > 0:
                        cross_features['mean_correlation'] = np.array([upper_triangle.mean()], dtype=np.float32)
                        cross_features['std_correlation'] = np.array([upper_triangle.std()], dtype=np.float32)
                        cross_features['max_correlation'] = np.array([upper_triangle.max()], dtype=np.float32)
                        cross_features['min_correlation'] = np.array([upper_triangle.min()], dtype=np.float32)
                        
                except Exception as e:
                    self.logger.error(f"Error calculating correlation matrix statistics: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error extracting cross-component features: {e}")
        
        return cross_features

    def _calculate_integration_quality(self, 
                                     components_data: Dict[int, ComponentDataExtract],
                                     aligned_time_series: pd.DataFrame) -> float:
        """Calculate quality score for component integration"""
        
        try:
            quality_factors = []
            
            # Data availability factor
            num_components_with_data = sum(1 for data in components_data.values() if len(data.primary_time_series) > 0)
            data_availability = num_components_with_data / len(components_data) if components_data else 0.0
            quality_factors.append(data_availability)
            
            # Time alignment quality
            if len(aligned_time_series) > 0:
                # Calculate percentage of non-null values
                total_cells = aligned_time_series.size
                non_null_cells = aligned_time_series.notna().sum().sum()
                alignment_quality = non_null_cells / total_cells if total_cells > 0 else 0.0
                quality_factors.append(alignment_quality)
            else:
                quality_factors.append(0.0)
            
            # Component confidence average
            avg_confidence = np.mean([data.confidence_score for data in components_data.values()])
            quality_factors.append(avg_confidence)
            
            # Processing time efficiency (inverse relationship)
            max_processing_time = max([data.processing_time_ms for data in components_data.values()] + [1.0])
            efficiency_score = max(0.0, 1.0 - (max_processing_time / 1000.0))  # Normalize to seconds
            quality_factors.append(efficiency_score)
            
            # Overall integration quality
            integration_quality = np.mean(quality_factors)
            return max(0.0, min(1.0, integration_quality))
            
        except Exception as e:
            self.logger.error(f"Error calculating integration quality: {e}")
            return 0.0


class ComponentIntegrationBridge:
    """Main bridge for integrating Component 6 with existing Components 1-5"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"vertex_mr.{self.__class__.__name__}")
        
        # Initialize sub-systems
        self.data_extractor = ComponentDataExtractor(config)
        self.data_aligner = ComponentDataAligner(config)
        
        # Integration settings
        self.required_components = config.get('required_components', [1, 2, 3, 4, 5])
        self.fallback_enabled = config.get('fallback_enabled', True)
        self.performance_logging = config.get('performance_logging', True)
        
        self.logger.info(f"Component Integration Bridge initialized for {len(self.required_components)} components")

    async def integrate_components(self, component_results: Dict[int, Any]) -> IntegratedComponentData:
        """
        Main integration method - processes all component results
        
        Args:
            component_results: Dict mapping component_id -> component result
            
        Returns:
            IntegratedComponentData with processed and aligned data
        """
        start_time = time.time()
        
        try:
            # Extract data from each component
            components_data = {}
            
            for component_id in self.required_components:
                if component_id in component_results:
                    component_data = self.data_extractor.extract_component_data(
                        component_id, component_results[component_id]
                    )
                    components_data[component_id] = component_data
                elif self.fallback_enabled:
                    # Create fallback data for missing component
                    fallback_data = self._create_fallback_component_data(component_id)
                    components_data[component_id] = fallback_data
            
            # Align and integrate data
            integrated_data = self.data_aligner.align_component_data(components_data)
            
            # Log performance if enabled
            if self.performance_logging:
                total_time = (time.time() - start_time) * 1000
                self._log_integration_performance(integrated_data, total_time)
            
            return integrated_data
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Component integration failed: {e}")
            
            # Return minimal integrated data on failure
            return IntegratedComponentData(
                components_data={},
                aligned_time_series=pd.DataFrame(),
                common_time_index=pd.DatetimeIndex([]),
                cross_component_features={},
                integration_quality_score=0.0,
                total_processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )

    def _create_fallback_component_data(self, component_id: int) -> ComponentDataExtract:
        """Create fallback data for missing component"""
        
        return ComponentDataExtract(
            component_id=component_id,
            component_name=f'Component_{component_id}_Fallback',
            primary_time_series=pd.DataFrame(),
            secondary_metrics={},
            feature_vector=np.zeros(20, dtype=np.float32),  # Default feature size
            confidence_score=0.0,
            processing_time_ms=0.0,
            timestamp=datetime.utcnow()
        )

    def _log_integration_performance(self, integrated_data: IntegratedComponentData, total_time: float):
        """Log integration performance metrics"""
        
        try:
            num_components = len(integrated_data.components_data)
            data_points = len(integrated_data.aligned_time_series)
            quality_score = integrated_data.integration_quality_score
            
            self.logger.info(
                f"Integration completed: {num_components} components, "
                f"{data_points} aligned data points, "
                f"quality={quality_score:.3f}, "
                f"time={total_time:.1f}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging integration performance: {e}")