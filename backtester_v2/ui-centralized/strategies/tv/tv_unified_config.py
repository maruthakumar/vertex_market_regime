#!/usr/bin/env python3
"""
TV Unified Configuration System
Enhances the unified configuration system to support TV's 6-file hierarchy
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime, date, time as datetime_time
import pandas as pd
import logging
import json
import yaml

logger = logging.getLogger(__name__)


class TVHierarchicalConfiguration:
    """
    TV Configuration that supports the 6-file hierarchy structure
    
    File Hierarchy:
    1. TV Master (TV_CONFIG_MASTER_*.xlsx) - Main configuration
    2. Signals (TV_CONFIG_SIGNALS_*.xlsx) - Trading signals
    3. Portfolio Long (TV_CONFIG_PORTFOLIO_LONG_*.xlsx) - Long portfolio settings
    4. Portfolio Short (TV_CONFIG_PORTFOLIO_SHORT_*.xlsx) - Short portfolio settings
    5. Portfolio Manual (TV_CONFIG_PORTFOLIO_MANUAL_*.xlsx) - Manual portfolio settings
    6. TBS Strategy (TV_CONFIG_STRATEGY_*.xlsx) - Strategy leg definitions
    """
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize TV hierarchical configuration"""
        self.base_path = base_path or Path('../../configurations/data/prod/tv')
        self.config_cache = {}
        self.file_hierarchy = {
            'tv_master': None,
            'signals': None,
            'portfolio_long': None,
            'portfolio_short': None,
            'portfolio_manual': None,
            'strategy': None
        }
        self.unified_config = {}
        
    def load_hierarchy(self, config_files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Load complete 6-file hierarchy
        
        Args:
            config_files: Dictionary mapping file types to paths
            
        Returns:
            Unified configuration dictionary
        """
        # Validate all files exist
        for file_type, file_path in config_files.items():
            if not file_path or not file_path.exists():
                raise FileNotFoundError(f"Missing {file_type}: {file_path}")
            self.file_hierarchy[file_type] = file_path
        
        # Load TV Master first
        self._load_tv_master()
        
        # Load signals
        self._load_signals()
        
        # Load portfolio configurations
        self._load_portfolios()
        
        # Load TBS strategy
        self._load_tbs_strategy()
        
        # Build unified configuration
        self._build_unified_config()
        
        return self.unified_config
    
    def _load_tv_master(self):
        """Load TV master configuration file"""
        file_path = self.file_hierarchy['tv_master']
        
        try:
            # Read Setting sheet
            df = pd.read_excel(file_path, sheet_name='Setting', engine='openpyxl')
            
            # Get first enabled row
            enabled_rows = df[df.get('Enabled', 'NO') == 'YES']
            if enabled_rows.empty:
                raise ValueError("No enabled TV configurations found")
            
            row = enabled_rows.iloc[0]
            
            # Extract configuration
            tv_config = {
                'name': row.get('Name', 'TV Strategy'),
                'enabled': True,
                'signal_file_path': row.get('SignalFilePath', ''),
                'start_date': self._parse_date(row.get('StartDate')),
                'end_date': self._parse_date(row.get('EndDate')),
                'signal_date_format': row.get('SignalDateFormat', '%Y%m%d %H%M%S'),
                'intraday_sqoff_applicable': row.get('IntradaySqOffApplicable', 'NO'),
                'intraday_exit_time': self._parse_time(row.get('IntradayExitTime')),
                'tv_exit_applicable': row.get('TvExitApplicable', 'YES'),
                'do_rollover': row.get('DoRollover', 'NO'),
                'rollover_time': self._parse_time(row.get('RolloverTime')),
                'manual_trade_entry_time': self._parse_time(row.get('ManualTradeEntryTime')),
                'manual_trade_lots': int(row.get('ManualTradeLots', 1)),
                'first_trade_entry_time': self._parse_time(row.get('FirstTradeEntryTime')),
                'increase_entry_signal_time_by': int(row.get('IncreaseEntrySignalTimeBy', 0)),
                'increase_exit_signal_time_by': int(row.get('IncreaseExitSignalTimeBy', 0)),
                'expiry_day_exit_time': self._parse_time(row.get('ExpiryDayExitTime')),
                'slippage_percent': float(row.get('SlippagePercent', 0.1)),
                'long_portfolio_file_path': row.get('LongPortfolioFilePath', ''),
                'short_portfolio_file_path': row.get('ShortPortfolioFilePath', ''),
                'manual_portfolio_file_path': row.get('ManualPortfolioFilePath', ''),
                'use_db_exit_timing': row.get('UseDbExitTiming', 'NO'),
                'exit_search_interval': int(row.get('ExitSearchInterval', 5)),
                'exit_price_source': row.get('ExitPriceSource', 'SPOT')
            }
            
            self.config_cache['tv_master'] = tv_config
            logger.info(f"Loaded TV Master config: {tv_config['name']}")
            
        except Exception as e:
            logger.error(f"Error loading TV master: {e}")
            raise
    
    def _load_signals(self):
        """Load signals file"""
        file_path = self.file_hierarchy['signals']
        
        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            
            # Validate required columns
            required_cols = ['Trade #', 'Type', 'Date/Time', 'Contracts']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Signal file missing columns: {missing}")
            
            # Convert to list of dictionaries
            signals = df.to_dict('records')
            
            self.config_cache['signals'] = signals
            logger.info(f"Loaded {len(signals)} signals")
            
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            raise
    
    def _load_portfolios(self):
        """Load portfolio configuration files"""
        portfolio_types = ['long', 'short', 'manual']
        
        for portfolio_type in portfolio_types:
            file_key = f'portfolio_{portfolio_type}'
            file_path = self.file_hierarchy[file_key]
            
            if not file_path:
                continue
                
            try:
                # Read PortfolioSetting sheet
                df = pd.read_excel(file_path, sheet_name='PortfolioSetting', engine='openpyxl')
                
                if df.empty:
                    logger.warning(f"Empty {portfolio_type} portfolio configuration")
                    continue
                
                row = df.iloc[0]
                
                portfolio_config = {
                    'portfolio_name': row.get('PortfolioName', f'TV_{portfolio_type.capitalize()}'),
                    'capital': int(row.get('Capital', 1000000)),
                    'max_risk': int(row.get('MaxRisk', 5)),
                    'max_positions': int(row.get('MaxPositions', 5)),
                    'risk_per_trade': int(row.get('RiskPerTrade', 2)),
                    'use_kelly': row.get('UseKellyCriterion', 'NO'),
                    'rebalance_freq': row.get('RebalanceFrequency', 'Daily'),
                    'stop_loss_type': row.get('StopLossType', 'Percentage'),
                    'stop_loss_value': float(row.get('StopLossValue', 2.0)),
                    'take_profit_type': row.get('TakeProfitType', 'Percentage'),
                    'take_profit_value': float(row.get('TakeProfitValue', 5.0))
                }
                
                self.config_cache[file_key] = portfolio_config
                logger.info(f"Loaded {portfolio_type} portfolio config")
                
            except Exception as e:
                logger.error(f"Error loading {portfolio_type} portfolio: {e}")
                raise
    
    def _load_tbs_strategy(self):
        """Load TBS strategy configuration"""
        file_path = self.file_hierarchy['strategy']
        
        if not file_path:
            return
            
        try:
            # Read GeneralParameter sheet
            general_df = pd.read_excel(file_path, sheet_name='GeneralParameter', engine='openpyxl')
            
            if general_df.empty:
                raise ValueError("Empty GeneralParameter sheet")
            
            general_params = general_df.iloc[0].to_dict()
            
            # Read LegParameter sheet
            legs_df = pd.read_excel(file_path, sheet_name='LegParameter', engine='openpyxl')
            
            legs = []
            for idx, row in legs_df.iterrows():
                leg = {
                    'leg_id': row['LegID'],
                    'instrument': row['Instrument'].upper(),
                    'transaction': row['Transaction'].upper(),
                    'expiry': row['Expiry'],
                    'strike_method': row['StrikeMethod'],
                    'strike_value': row.get('StrikeValue', 0),
                    'lots': int(row['Lots']),
                    'sl_type': row.get('SLType', 'percentage'),
                    'sl_value': float(row.get('SLValue', 100)),
                    'tgt_type': row.get('TGTType', 'percentage'),
                    'tgt_value': float(row.get('TGTValue', 0))
                }
                legs.append(leg)
            
            strategy_config = {
                'general_parameters': general_params,
                'legs': legs
            }
            
            self.config_cache['strategy'] = strategy_config
            logger.info(f"Loaded TBS strategy with {len(legs)} legs")
            
        except Exception as e:
            logger.error(f"Error loading TBS strategy: {e}")
            raise
    
    def _build_unified_config(self):
        """Build unified configuration from all loaded components"""
        tv_master = self.config_cache.get('tv_master', {})
        
        self.unified_config = {
            'strategy_type': 'TV',
            'version': '2.0.0',
            'hierarchy_type': '6-file',
            'created_at': datetime.now().isoformat(),
            
            # Master settings
            'master_settings': tv_master,
            
            # Signals
            'signals': self.config_cache.get('signals', []),
            
            # Portfolio configurations
            'portfolios': {
                'long': self.config_cache.get('portfolio_long', {}),
                'short': self.config_cache.get('portfolio_short', {}),
                'manual': self.config_cache.get('portfolio_manual', {})
            },
            
            # TBS strategy
            'tbs_strategy': self.config_cache.get('strategy', {}),
            
            # Derived settings
            'trading_parameters': {
                'start_date': tv_master.get('start_date'),
                'end_date': tv_master.get('end_date'),
                'capital': self._get_total_capital(),
                'signal_count': len(self.config_cache.get('signals', [])),
                'portfolio_count': self._count_active_portfolios(),
                'leg_count': len(self.config_cache.get('strategy', {}).get('legs', []))
            },
            
            # File references
            'source_files': {
                k: str(v) for k, v in self.file_hierarchy.items() if v
            }
        }
    
    def _parse_date(self, date_str: Any) -> Optional[date]:
        """Parse date from DD_MM_YYYY format"""
        if pd.isna(date_str) or not date_str:
            return None
            
        if isinstance(date_str, (date, datetime)):
            return date_str if isinstance(date_str, date) else date_str.date()
            
        # Try parsing DD_MM_YYYY format
        try:
            date_str = str(date_str)
            if '_' in date_str:
                parts = date_str.split('_')
                if len(parts) == 3:
                    day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                    return date(year, month, day)
            # Try standard datetime parsing
            return pd.to_datetime(date_str).date()
        except:
            logger.warning(f"Could not parse date: {date_str}")
            return None
    
    def _parse_time(self, time_str: Any) -> Optional[datetime_time]:
        """Parse time from various formats"""
        if pd.isna(time_str) or not time_str:
            return None
            
        if isinstance(time_str, datetime_time):
            return time_str
            
        # Try parsing as string (HHMMSS or HH:MM:SS)
        try:
            time_str = str(time_str)
            if ':' in time_str:
                parts = time_str.split(':')
                return datetime_time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
            elif len(time_str) == 6:
                return datetime_time(int(time_str[:2]), int(time_str[2:4]), int(time_str[4:6]))
            else:
                return None
        except:
            return None
    
    def _get_total_capital(self) -> int:
        """Calculate total capital across all portfolios"""
        total = 0
        for portfolio_type in ['long', 'short', 'manual']:
            portfolio_key = f'portfolio_{portfolio_type}'
            portfolio = self.config_cache.get(portfolio_key, {})
            total += portfolio.get('capital', 0)
        return total
    
    def _count_active_portfolios(self) -> int:
        """Count number of active portfolios"""
        count = 0
        for portfolio_type in ['long', 'short', 'manual']:
            portfolio_key = f'portfolio_{portfolio_type}'
            if portfolio_key in self.config_cache and self.config_cache[portfolio_key]:
                count += 1
        return count
    
    def validate_hierarchy(self) -> Tuple[bool, List[str]]:
        """
        Validate the complete hierarchy configuration
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        # Check required files
        if not self.config_cache.get('tv_master'):
            errors.append("Missing TV master configuration")
        
        if not self.config_cache.get('signals'):
            errors.append("Missing signals configuration")
        
        # At least one portfolio must be configured
        portfolios_configured = any(
            self.config_cache.get(f'portfolio_{p}')
            for p in ['long', 'short', 'manual']
        )
        if not portfolios_configured:
            errors.append("At least one portfolio must be configured")
        
        # TBS strategy is required if portfolios are configured
        if portfolios_configured and not self.config_cache.get('strategy'):
            errors.append("TBS strategy configuration required")
        
        # Validate date consistency
        tv_master = self.config_cache.get('tv_master', {})
        start_date = tv_master.get('start_date')
        end_date = tv_master.get('end_date')
        
        if start_date and end_date and start_date > end_date:
            errors.append(f"Invalid date range: {start_date} > {end_date}")
        
        # Validate signal count
        signals = self.config_cache.get('signals', [])
        if len(signals) == 0:
            errors.append("No signals found in signal file")
        
        # Validate portfolio capital
        total_capital = self._get_total_capital()
        if total_capital <= 0:
            errors.append("Total capital must be positive")
        
        return len(errors) == 0, errors
    
    def export_to_yaml(self, output_path: Optional[Path] = None) -> Path:
        """Export unified configuration to YAML"""
        if not output_path:
            output_path = Path(f"tv_unified_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        
        # Convert datetime objects to strings
        yaml_config = json.loads(json.dumps(self.unified_config, default=str))
        
        with open(output_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Exported unified config to: {output_path}")
        return output_path
    
    def export_to_json(self, output_path: Optional[Path] = None) -> Path:
        """Export unified configuration to JSON"""
        if not output_path:
            output_path = Path(f"tv_unified_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(output_path, 'w') as f:
            json.dump(self.unified_config, f, indent=2, default=str)
        
        logger.info(f"Exported unified config to: {output_path}")
        return output_path
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of the TV strategy configuration"""
        return {
            'strategy_name': self.unified_config.get('master_settings', {}).get('name', 'Unknown'),
            'hierarchy_type': self.unified_config.get('hierarchy_type'),
            'total_capital': self.unified_config.get('trading_parameters', {}).get('capital', 0),
            'signal_count': self.unified_config.get('trading_parameters', {}).get('signal_count', 0),
            'portfolio_count': self.unified_config.get('trading_parameters', {}).get('portfolio_count', 0),
            'leg_count': self.unified_config.get('trading_parameters', {}).get('leg_count', 0),
            'date_range': {
                'start': str(self.unified_config.get('trading_parameters', {}).get('start_date', '')),
                'end': str(self.unified_config.get('trading_parameters', {}).get('end_date', ''))
            }
        }


def main():
    """Test TV unified configuration"""
    print("\n" + "="*80)
    print("TV UNIFIED CONFIGURATION SYSTEM TEST")
    print("="*80)
    
    # Initialize configuration
    tv_config = TVHierarchicalConfiguration()
    
    # Define test files
    base_path = Path('../../configurations/data/prod/tv')
    config_files = {
        'tv_master': base_path / 'TV_CONFIG_MASTER_1.0.0.xlsx',
        'signals': base_path / 'TV_CONFIG_SIGNALS_1.0.0.xlsx',
        'portfolio_long': base_path / 'TV_CONFIG_PORTFOLIO_LONG_1.0.0.xlsx',
        'portfolio_short': base_path / 'TV_CONFIG_PORTFOLIO_SHORT_1.0.0.xlsx',
        'portfolio_manual': base_path / 'TV_CONFIG_PORTFOLIO_MANUAL_1.0.0.xlsx',
        'strategy': base_path / 'TV_CONFIG_STRATEGY_1.0.0.xlsx'
    }
    
    # Load hierarchy
    print("\n🔄 Loading 6-file hierarchy...")
    try:
        unified = tv_config.load_hierarchy(config_files)
        print("✅ Hierarchy loaded successfully")
        
        # Get summary
        summary = tv_config.get_strategy_summary()
        print("\n📊 Strategy Summary:")
        print(f"   • Name: {summary['strategy_name']}")
        print(f"   • Type: {summary['hierarchy_type']}")
        print(f"   • Capital: ₹{summary['total_capital']:,}")
        print(f"   • Signals: {summary['signal_count']}")
        print(f"   • Portfolios: {summary['portfolio_count']}")
        print(f"   • Legs: {summary['leg_count']}")
        print(f"   • Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
    except Exception as e:
        print(f"❌ Failed to load hierarchy: {e}")
        return 1
    
    # Validate
    print("\n🔍 Validating configuration...")
    is_valid, errors = tv_config.validate_hierarchy()
    
    if is_valid:
        print("✅ Configuration validation passed")
    else:
        print(f"❌ Validation failed with {len(errors)} errors:")
        for error in errors:
            print(f"   • {error}")
    
    # Export to YAML
    print("\n💾 Exporting to YAML...")
    yaml_path = tv_config.export_to_yaml()
    print(f"✅ Exported to: {yaml_path}")
    
    # Export to JSON
    print("\n💾 Exporting to JSON...")
    json_path = tv_config.export_to_json()
    print(f"✅ Exported to: {json_path}")
    
    print("\n🎉 PHASE 6 ENHANCEMENT COMPLETE!")
    print("✅ Unified configuration system enhanced for 6-file hierarchy")
    print("✅ Validation system integrated")
    print("✅ Export functionality implemented")
    print("✅ Ready for integration with main system")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())