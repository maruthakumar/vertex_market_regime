"""Enhanced OI parser with full backward compatibility for legacy formats."""

import pandas as pd
import os
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import logging

from .enhanced_models import (
    EnhancedOIConfig, EnhancedLegConfig, DynamicWeightConfig, 
    FactorConfig, PortfolioConfig, StrategyConfig, LegacyConfig,
    OiThresholdType, StrikeRangeType, NormalizationMethod, OutlierHandling
)

logger = logging.getLogger(__name__)

class EnhancedOIParser:
    """Enhanced OI parser with full backward compatibility."""
    
    def __init__(self):
        """Initialize the enhanced parser."""
        self.supported_formats = ['legacy', 'enhanced', 'hybrid']
        
    def detect_format(self, file_path: str) -> str:
        """Detect the input file format."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
                
            xl_file = pd.ExcelFile(file_path)
            sheets = xl_file.sheet_names
            
            # Check for enhanced format
            if all(sheet in sheets for sheet in ['GeneralParameter', 'LegParameter', 'WeightConfiguration', 'FactorParameters']):
                return 'enhanced'
            
            # Check for enhanced portfolio format
            if all(sheet in sheets for sheet in ['PortfolioSetting', 'StrategySetting']):
                return 'enhanced_portfolio'
                
            # Check for legacy bt_setting format
            if all(sheet in sheets for sheet in ['MainSetting', 'Strategy No Info']):
                return 'legacy_bt_setting'
                
            # Check for legacy maxoi format
            if 'Sheet1' in sheets:
                df = pd.read_excel(file_path, sheet_name='Sheet1')
                legacy_columns = ['id', 'underlyingname', 'startdate', 'enddate', 'entrytime']
                if all(col in df.columns for col in legacy_columns):
                    return 'legacy_maxoi'
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Error detecting format for {file_path}: {e}")
            return 'unknown'
    
    def parse_enhanced_config(self, file_path: str) -> Tuple[List[EnhancedOIConfig], List[EnhancedLegConfig], DynamicWeightConfig, List[FactorConfig]]:
        """Parse enhanced OI configuration file with 4 sheets."""
        try:
            # Read all sheets
            general_df = pd.read_excel(file_path, sheet_name='GeneralParameter')
            leg_df = pd.read_excel(file_path, sheet_name='LegParameter')
            weight_df = pd.read_excel(file_path, sheet_name='WeightConfiguration')
            factor_df = pd.read_excel(file_path, sheet_name='FactorParameters')
            
            # Parse general parameters
            general_configs = []
            for _, row in general_df.iterrows():
                config = EnhancedOIConfig(
                    strategy_name=row.get('StrategyName', 'Enhanced_Strategy'),
                    underlying=row.get('Underlying', 'SPOT'),
                    index=row.get('Index', 'NIFTY'),
                    dte=row.get('DTE', 0),
                    timeframe=row.get('Timeframe', 3),
                    start_time=self._format_time(row.get('StartTime', '091600')),
                    end_time=self._format_time(row.get('EndTime', '152000')),
                    last_entry_time=self._format_time(row.get('LastEntryTime', '150000')),
                    strike_selection_time=self._format_time(row.get('StrikeSelectionTime', '091530')),
                    max_open_positions=row.get('MaxOpenPositions', 2),
                    oi_threshold=row.get('OiThreshold', 800000),
                    strike_count=row.get('StrikeCount', 10),
                    weekdays=row.get('Weekdays', '1,2,3,4,5'),
                    strategy_profit=row.get('StrategyProfit', 0),
                    strategy_loss=row.get('StrategyLoss', 0),
                    
                    # OI-Specific Parameters
                    oi_method=row.get('OiMethod', 'MAXOI_1'),
                    coi_based_on=row.get('CoiBasedOn', 'YESTERDAY_CLOSE'),
                    oi_recheck_interval=row.get('OiRecheckInterval', 180),
                    oi_threshold_type=OiThresholdType(row.get('OiThresholdType', 'ABSOLUTE')),
                    oi_concentration_threshold=row.get('OiConcentrationThreshold', 0.3),
                    oi_distribution_analysis=row.get('OiDistributionAnalysis', 'YES') == 'YES',
                    strike_range_type=StrikeRangeType(row.get('StrikeRangeType', 'FIXED')),
                    strike_range_value=row.get('StrikeRangeValue', 10),
                    oi_momentum_period=row.get('OiMomentumPeriod', 5),
                    oi_trend_analysis=row.get('OiTrendAnalysis', 'YES') == 'YES',
                    oi_seasonal_adjustment=row.get('OiSeasonalAdjustment', 'NO') == 'YES',
                    oi_volume_correlation=row.get('OiVolumeCorrelation', 'YES') == 'YES',
                    oi_liquidity_filter=row.get('OiLiquidityFilter', 'YES') == 'YES',
                    oi_anomaly_detection=row.get('OiAnomalyDetection', 'YES') == 'YES',
                    oi_signal_confirmation=row.get('OiSignalConfirmation', 'YES') == 'YES',
                    
                    # Dynamic Weightage Parameters
                    enable_dynamic_weights=row.get('EnableDynamicWeights', 'YES') == 'YES',
                    weight_adjustment_period=row.get('WeightAdjustmentPeriod', 20),
                    learning_rate=row.get('LearningRate', 0.01),
                    performance_window=row.get('PerformanceWindow', 100),
                    min_weight=row.get('MinWeight', 0.05),
                    max_weight=row.get('MaxWeight', 0.50),
                    weight_decay_factor=row.get('WeightDecayFactor', 0.95),
                    correlation_threshold=row.get('CorrelationThreshold', 0.7),
                    diversification_bonus=row.get('DiversificationBonus', 1.1),
                    regime_adjustment=row.get('RegimeAdjustment', 'YES') == 'YES',
                    volatility_adjustment=row.get('VolatilityAdjustment', 'YES') == 'YES',
                    liquidity_adjustment=row.get('LiquidityAdjustment', 'YES') == 'YES',
                    trend_adjustment=row.get('TrendAdjustment', 'YES') == 'YES',
                    momentum_adjustment=row.get('MomentumAdjustment', 'YES') == 'YES',
                    seasonal_adjustment=row.get('SeasonalAdjustment', 'NO') == 'YES'
                )
                general_configs.append(config)
            
            # Parse leg parameters
            leg_configs = []
            for _, row in leg_df.iterrows():
                config = EnhancedLegConfig(
                    strategy_name=row.get('StrategyName', 'Enhanced_Strategy'),
                    leg_id=row.get('LegID', 'LEG_1'),
                    instrument=row.get('Instrument', 'CE'),
                    transaction=row.get('Transaction', 'SELL'),
                    expiry=row.get('Expiry', 'current'),
                    strike_method=row.get('StrikeMethod', 'MAXOI_1'),
                    strike_value=row.get('StrikeValue', 0),
                    lots=row.get('Lots', 1),
                    sl_type=row.get('SLType', 'percentage'),
                    sl_value=row.get('SLValue', 30),
                    tgt_type=row.get('TGTType', 'percentage'),
                    tgt_value=row.get('TGTValue', 50),
                    trail_sl_type=row.get('TrailSLType', 'percentage'),
                    sl_trail_at=row.get('SL_TrailAt', 25),
                    sl_trail_by=row.get('SL_TrailBy', 10),
                    
                    # OI-Specific Leg Parameters
                    leg_oi_threshold=row.get('LegOiThreshold', 800000),
                    leg_oi_weight=row.get('LegOiWeight', 0.4),
                    leg_coi_weight=row.get('LegCoiWeight', 0.3),
                    leg_oi_rank=row.get('LegOiRank', 1),
                    leg_oi_concentration=row.get('LegOiConcentration', 0.3),
                    leg_oi_momentum=row.get('LegOiMomentum', 0.2),
                    leg_oi_trend=row.get('LegOiTrend', 0.15),
                    leg_oi_liquidity=row.get('LegOiLiquidity', 0.1),
                    leg_oi_anomaly=row.get('LegOiAnomaly', 0.05),
                    leg_oi_confirmation=row.get('LegOiConfirmation', 'YES') == 'YES',
                    
                    # Greek-Based Parameters
                    delta_weight=row.get('DeltaWeight', 0.25),
                    gamma_weight=row.get('GammaWeight', 0.20),
                    theta_weight=row.get('ThetaWeight', 0.15),
                    vega_weight=row.get('VegaWeight', 0.10),
                    delta_threshold=row.get('DeltaThreshold', 0.5),
                    gamma_threshold=row.get('GammaThreshold', 0.1),
                    theta_threshold=row.get('ThetaThreshold', -0.05),
                    vega_threshold=row.get('VegaThreshold', 0.2),
                    greek_rebalance_freq=row.get('GreekRebalanceFreq', 300),
                    greek_risk_limit=row.get('GreekRiskLimit', 10000)
                )
                leg_configs.append(config)
            
            # Parse weight configuration
            weight_config = None
            if not weight_df.empty:
                row = weight_df.iloc[0]
                weight_config = DynamicWeightConfig(
                    oi_factor_weight=row.get('OiFactorWeight', 0.35),
                    coi_factor_weight=row.get('CoiFactorWeight', 0.25),
                    greek_factor_weight=row.get('GreekFactorWeight', 0.20),
                    market_factor_weight=row.get('MarketFactorWeight', 0.15),
                    performance_factor_weight=row.get('PerformanceFactorWeight', 0.05),
                    current_oi_weight=row.get('CurrentOiWeight', 0.30),
                    oi_concentration_weight=row.get('OiConcentrationWeight', 0.20),
                    oi_distribution_weight=row.get('OiDistributionWeight', 0.15),
                    oi_momentum_weight=row.get('OiMomentumWeight', 0.15),
                    oi_trend_weight=row.get('OiTrendWeight', 0.10),
                    oi_seasonal_weight=row.get('OiSeasonalWeight', 0.05),
                    oi_liquidity_weight=row.get('OiLiquidityWeight', 0.03),
                    oi_anomaly_weight=row.get('OiAnomalyWeight', 0.02),
                    weight_learning_rate=row.get('WeightLearningRate', 0.01),
                    weight_decay_factor=row.get('WeightDecayFactor', 0.95),
                    weight_smoothing_factor=row.get('WeightSmoothingFactor', 0.15),
                    min_factor_weight=row.get('MinFactorWeight', 0.05),
                    max_factor_weight=row.get('MaxFactorWeight', 0.50),
                    weight_rebalance_freq=row.get('WeightRebalanceFreq', 300),
                    performance_threshold=row.get('PerformanceThreshold', 0.6),
                    correlation_threshold=row.get('CorrelationThreshold', 0.7),
                    diversification_bonus=row.get('DiversificationBonus', 1.1),
                    volatility_adjustment=row.get('VolatilityAdjustment', 1.15),
                    trend_adjustment=row.get('TrendAdjustment', 1.1),
                    regime_adjustment=row.get('RegimeAdjustment', 1.2)
                )
            
            # Parse factor configurations
            factor_configs = []
            for _, row in factor_df.iterrows():
                config = FactorConfig(
                    factor_name=row.get('FactorName', 'UNKNOWN'),
                    factor_type=row.get('FactorType', 'OI'),
                    base_weight=row.get('BaseWeight', 0.1),
                    min_weight=row.get('MinWeight', 0.05),
                    max_weight=row.get('MaxWeight', 0.50),
                    lookback_period=row.get('LookbackPeriod', 5),
                    smoothing_factor=row.get('SmoothingFactor', 0.2),
                    threshold_type=row.get('ThresholdType', 'ABSOLUTE'),
                    threshold_value=row.get('ThresholdValue', 0),
                    normalization_method=NormalizationMethod(row.get('NormalizationMethod', 'ZSCORE')),
                    outlier_handling=OutlierHandling(row.get('OutlierHandling', 'WINSORIZE')),
                    seasonal_adjustment=row.get('SeasonalAdjustment', 'NO') == 'YES',
                    volatility_adjustment=row.get('VolatilityAdjustment', 'YES') == 'YES',
                    trend_adjustment=row.get('TrendAdjustment', 'YES') == 'YES',
                    regime_adjustment=row.get('RegimeAdjustment', 'YES') == 'YES',
                    performance_tracking=row.get('PerformanceTracking', 'YES') == 'YES'
                )
                factor_configs.append(config)
            
            return general_configs, leg_configs, weight_config, factor_configs
            
        except Exception as e:
            logger.error(f"Error parsing enhanced config: {e}")
            raise
    
    def parse_enhanced_portfolio(self, file_path: str) -> Tuple[PortfolioConfig, List[StrategyConfig]]:
        """Parse enhanced portfolio configuration."""
        try:
            portfolio_df = pd.read_excel(file_path, sheet_name='PortfolioSetting')
            strategy_df = pd.read_excel(file_path, sheet_name='StrategySetting')
            
            # Parse portfolio configuration
            portfolio_config = None
            if not portfolio_df.empty:
                row = portfolio_df.iloc[0]
                portfolio_config = PortfolioConfig(
                    start_date=row.get('StartDate', '01_01_2025'),
                    end_date=row.get('EndDate', '31_12_2025'),
                    is_tick_bt=row.get('IsTickBT', 'NO') == 'YES',
                    enabled=row.get('Enabled', 'YES') == 'YES',
                    portfolio_name=row.get('PortfolioName', 'OI_Enhanced_Portfolio'),
                    portfolio_target=row.get('PortfolioTarget', 0),
                    portfolio_stoploss=row.get('PortfolioStoploss', 0),
                    portfolio_trailing_type=row.get('PortfolioTrailingType', 'Lock Minimum Profit'),
                    pnl_cal_time=self._format_time(row.get('PnLCalTime', '151500')),
                    lock_percent=row.get('LockPercent', 0),
                    trail_percent=row.get('TrailPercent', 0),
                    sq_off1_time=self._format_time(row.get('SqOff1Time', '000000')),
                    sq_off1_percent=row.get('SqOff1Percent', 0),
                    sq_off2_time=self._format_time(row.get('SqOff2Time', '000000')),
                    sq_off2_percent=row.get('SqOff2Percent', 0),
                    profit_reaches=row.get('ProfitReaches', 0),
                    lock_min_profit_at=row.get('LockMinProfitAt', 0),
                    increase_in_profit=row.get('IncreaseInProfit', 0),
                    trail_min_profit_by=row.get('TrailMinProfitBy', 0),
                    multiplier=row.get('Multiplier', 1),
                    slippage_percent=row.get('SlippagePercent', 0.1)
                )
            
            # Parse strategy configurations
            strategy_configs = []
            for _, row in strategy_df.iterrows():
                config = StrategyConfig(
                    enabled=row.get('Enabled', 'YES') == 'YES',
                    portfolio_name=row.get('PortfolioName', 'OI_Enhanced_Portfolio'),
                    strategy_type=row.get('StrategyType', 'OI'),
                    strategy_excel_file_path=row.get('StrategyExcelFilePath', '')
                )
                strategy_configs.append(config)
            
            return portfolio_config, strategy_configs
            
        except Exception as e:
            logger.error(f"Error parsing enhanced portfolio: {e}")
            raise
    
    def _format_time(self, time_value: Union[str, int, float]) -> str:
        """Format time value to HHMMSS string."""
        if isinstance(time_value, (int, float)):
            return str(int(time_value)).zfill(6)
        return str(time_value).zfill(6)
    
    def parse_legacy_bt_setting(self, file_path: str) -> Tuple[List[StrategyConfig], List[str]]:
        """Parse legacy bt_setting.xlsx file."""
        try:
            main_df = pd.read_excel(file_path, sheet_name='MainSetting')
            info_df = pd.read_excel(file_path, sheet_name='Strategy No Info')

            strategy_configs = []
            strategy_names = []

            for _, row in main_df.iterrows():
                config = StrategyConfig(
                    enabled=row.get('enabled', 'yes').lower() == 'yes',
                    portfolio_name=f'Legacy_Portfolio_{row.get("stgyno", 1)}',
                    strategy_type='OI_LEGACY',
                    strategy_excel_file_path=row.get('stgyfilepath', '')
                )
                strategy_configs.append(config)

            for _, row in info_df.iterrows():
                strategy_names.append(row.get('Strategy Name', 'Legacy Strategy'))

            return strategy_configs, strategy_names

        except Exception as e:
            logger.error(f"Error parsing legacy bt_setting: {e}")
            raise

    def parse_legacy_maxoi(self, file_path: str) -> List[EnhancedOIConfig]:
        """Parse legacy input_maxoi.xlsx file and convert to enhanced format."""
        try:
            maxoi_df = pd.read_excel(file_path, sheet_name='Sheet1')

            enhanced_configs = []

            for _, row in maxoi_df.iterrows():
                # Convert legacy parameters to enhanced format with conservative defaults
                config = EnhancedOIConfig(
                    # Core Strategy Parameters (mapped from legacy)
                    strategy_name=row.get('id', 'Legacy_Strategy'),
                    underlying='SPOT',  # Default
                    index=row.get('underlyingname', 'NIFTY'),
                    dte=row.get('dte', 0),
                    timeframe=3,  # Default timeframe
                    start_time=self._format_time(row.get('entrytime', 92200)),
                    end_time=self._format_time(row.get('exittime', 152500)),
                    last_entry_time=self._format_time(row.get('lastentrytime', 152400)),
                    strike_selection_time='091530',  # Default
                    max_open_positions=2,  # Default
                    oi_threshold=800000,  # Default
                    strike_count=row.get('noofstrikeeachside', 40),
                    weekdays='1,2,3,4,5',  # Default
                    strategy_profit=row.get('strategymaxprofit', 0),
                    strategy_loss=row.get('strategymaxloss', 0),

                    # OI-Specific Parameters (conservative defaults for legacy)
                    oi_method=self._convert_legacy_strike_method(row.get('striketotrade', 'maxoi1')),
                    coi_based_on='YESTERDAY_CLOSE',  # Default
                    oi_recheck_interval=180,  # Default
                    oi_threshold_type=OiThresholdType.ABSOLUTE,  # Default
                    oi_concentration_threshold=0.3,  # Default
                    oi_distribution_analysis=False,  # Conservative for legacy
                    strike_range_type=StrikeRangeType.FIXED,  # Default
                    strike_range_value=row.get('noofstrikeeachside', 40),
                    oi_momentum_period=5,  # Default
                    oi_trend_analysis=False,  # Conservative for legacy
                    oi_seasonal_adjustment=False,  # Default
                    oi_volume_correlation=False,  # Conservative for legacy
                    oi_liquidity_filter=False,  # Conservative for legacy
                    oi_anomaly_detection=False,  # Conservative for legacy
                    oi_signal_confirmation=False,  # Conservative for legacy

                    # Dynamic Weightage Parameters (disabled for legacy compatibility)
                    enable_dynamic_weights=False,  # Disabled for legacy
                    weight_adjustment_period=20,  # Default
                    learning_rate=0.01,  # Default
                    performance_window=100,  # Default
                    min_weight=0.05,  # Default
                    max_weight=0.50,  # Default
                    weight_decay_factor=0.95,  # Default
                    correlation_threshold=0.7,  # Default
                    diversification_bonus=1.1,  # Default
                    regime_adjustment=False,  # Disabled for legacy
                    volatility_adjustment=False,  # Disabled for legacy
                    liquidity_adjustment=False,  # Disabled for legacy
                    trend_adjustment=False,  # Disabled for legacy
                    momentum_adjustment=False,  # Disabled for legacy
                    seasonal_adjustment=False  # Disabled for legacy
                )
                enhanced_configs.append(config)

            return enhanced_configs

        except Exception as e:
            logger.error(f"Error parsing legacy maxoi: {e}")
            raise

    def create_legacy_leg_configs(self, strategy_config: EnhancedOIConfig) -> List[EnhancedLegConfig]:
        """Create default leg configurations for legacy strategies."""
        leg_configs = []

        # Create CE leg
        ce_leg = EnhancedLegConfig(
            strategy_name=strategy_config.strategy_name,
            leg_id='CE_LEG_1',
            instrument='CE',
            transaction='SELL',
            expiry='current',
            strike_method=strategy_config.oi_method,
            strike_value=0,
            lots=1,  # Default
            sl_type='percentage',
            sl_value=30,  # Default
            tgt_type='percentage',
            tgt_value=50,  # Default
            trail_sl_type='percentage',
            sl_trail_at=25,  # Default
            sl_trail_by=10,  # Default

            # Conservative OI parameters for legacy
            leg_oi_threshold=strategy_config.oi_threshold,
            leg_oi_weight=1.0,  # Full weight for legacy
            leg_coi_weight=0.0,  # Disabled for legacy
            leg_oi_rank=1,
            leg_oi_concentration=0.0,  # Disabled for legacy
            leg_oi_momentum=0.0,  # Disabled for legacy
            leg_oi_trend=0.0,  # Disabled for legacy
            leg_oi_liquidity=0.0,  # Disabled for legacy
            leg_oi_anomaly=0.0,  # Disabled for legacy
            leg_oi_confirmation=False,  # Disabled for legacy

            # Disabled Greeks for legacy
            delta_weight=0.0,
            gamma_weight=0.0,
            theta_weight=0.0,
            vega_weight=0.0,
            delta_threshold=0.5,
            gamma_threshold=0.1,
            theta_threshold=-0.05,
            vega_threshold=0.2,
            greek_rebalance_freq=300,
            greek_risk_limit=10000
        )
        leg_configs.append(ce_leg)

        # Create PE leg if needed (based on strategy method)
        if 'MAXOI' in strategy_config.oi_method or 'MAXCOI' in strategy_config.oi_method:
            pe_leg = EnhancedLegConfig(
                strategy_name=strategy_config.strategy_name,
                leg_id='PE_LEG_1',
                instrument='PE',
                transaction='SELL',
                expiry='current',
                strike_method=strategy_config.oi_method,
                strike_value=0,
                lots=1,
                sl_type='percentage',
                sl_value=30,
                tgt_type='percentage',
                tgt_value=50,
                trail_sl_type='percentage',
                sl_trail_at=25,
                sl_trail_by=10,

                # Conservative OI parameters for legacy
                leg_oi_threshold=strategy_config.oi_threshold,
                leg_oi_weight=1.0,
                leg_coi_weight=0.0,
                leg_oi_rank=1,
                leg_oi_concentration=0.0,
                leg_oi_momentum=0.0,
                leg_oi_trend=0.0,
                leg_oi_liquidity=0.0,
                leg_oi_anomaly=0.0,
                leg_oi_confirmation=False,

                # Disabled Greeks for legacy
                delta_weight=0.0,
                gamma_weight=0.0,
                theta_weight=0.0,
                vega_weight=0.0,
                delta_threshold=-0.5,  # Negative for PE
                gamma_threshold=0.1,
                theta_threshold=-0.05,
                vega_threshold=0.2,
                greek_rebalance_freq=300,
                greek_risk_limit=10000
            )
            leg_configs.append(pe_leg)

        return leg_configs

    def _convert_legacy_strike_method(self, legacy_method: str) -> str:
        """Convert legacy strike method to enhanced OI method."""
        method_mapping = {
            'maxoi1': 'MAXOI_1',
            'maxoi2': 'MAXOI_2',
            'maxoi3': 'MAXOI_3',
            'maxoi4': 'MAXOI_4',
            'maxoi5': 'MAXOI_5',
            'maxcoi1': 'MAXCOI_1',
            'maxcoi2': 'MAXCOI_2',
            'maxcoi3': 'MAXCOI_3',
            'maxcoi4': 'MAXCOI_4',
            'maxcoi5': 'MAXCOI_5'
        }
        return method_mapping.get(legacy_method.lower(), 'MAXOI_1')
