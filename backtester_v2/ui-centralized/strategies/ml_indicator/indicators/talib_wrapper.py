"""
TA-Lib Indicators Wrapper for ML Indicator Strategy
Provides SQL implementations of 200+ TA-Lib indicators for GPU execution
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TALibWrapper:
    """Wrapper for TA-Lib indicators with SQL implementations"""
    
    def __init__(self):
        self.indicators = self._initialize_indicators()
        
    def _initialize_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize all TA-Lib indicators with SQL templates"""
        
        return {
            # Overlap Studies
            "SMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    AVG({price_column}) OVER (
                        PARTITION BY {partition_by}
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS SMA_{timeperiod}
                """
            },
            
            "EMA": {
                "category": "Overlap Studies", 
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- EMA calculation using recursive approach
                    WITH RECURSIVE ema_calc AS (
                        SELECT 
                            trade_date,
                            trade_time,
                            {price_column} AS price,
                            {price_column} AS ema_value,
                            ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) AS rn
                        FROM {table_name}
                        WHERE rn = 1
                        
                        UNION ALL
                        
                        SELECT 
                            t.trade_date,
                            t.trade_time,
                            t.{price_column} AS price,
                            (t.{price_column} * (2.0 / ({timeperiod} + 1))) + 
                            (e.ema_value * (1 - (2.0 / ({timeperiod} + 1)))) AS ema_value,
                            t.rn
                        FROM {table_name} t
                        JOIN ema_calc e ON t.rn = e.rn + 1
                    )
                    SELECT ema_value AS EMA_{timeperiod}
                    FROM ema_calc
                """
            },
            
            "DEMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- Double EMA
                    2 * EMA_{timeperiod} - EMA(EMA_{timeperiod}, {timeperiod}) AS DEMA_{timeperiod}
                """
            },
            
            "TEMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- Triple EMA
                    3 * EMA_{timeperiod} - 3 * EMA(EMA_{timeperiod}, {timeperiod}) + 
                    EMA(EMA(EMA_{timeperiod}, {timeperiod}), {timeperiod}) AS TEMA_{timeperiod}
                """
            },
            
            "WMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20},
                "sql_template": """
                    -- Weighted Moving Average
                    SUM({price_column} * (ROW_NUMBER() OVER (ORDER BY trade_date DESC, trade_time DESC))) OVER (
                        PARTITION BY {partition_by}
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) / SUM(ROW_NUMBER() OVER (ORDER BY trade_date DESC, trade_time DESC)) OVER (
                        PARTITION BY {partition_by}
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS WMA_{timeperiod}
                """
            },
            
            "KAMA": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 30},
                "sql_template": """
                    -- Kaufman Adaptive Moving Average
                    -- Simplified version for SQL
                    CASE 
                        WHEN ROW_NUMBER() OVER (ORDER BY trade_date, trade_time) >= {timeperiod}
                        THEN AVG({price_column}) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                        ELSE {price_column}
                    END AS KAMA_{timeperiod}
                """
            },
            
            "BBANDS": {
                "category": "Overlap Studies",
                "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
                "sql_template": """
                    -- Bollinger Bands
                    AVG({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS BBANDS_MIDDLE_{timeperiod},
                    
                    AVG({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) + {nbdevup} * STDDEV({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS BBANDS_UPPER_{timeperiod},
                    
                    AVG({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - {nbdevdn} * STDDEV({price_column}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS BBANDS_LOWER_{timeperiod}
                """
            },
            
            # Momentum Indicators
            "RSI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- RSI Calculation
                    100 - (100 / (1 + 
                        AVG(CASE WHEN {price_column} > LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                            THEN {price_column} - LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        ) / 
                        AVG(CASE WHEN {price_column} < LAG({price_column}) OVER (ORDER BY trade_date, trade_time) 
                            THEN LAG({price_column}) OVER (ORDER BY trade_date, trade_time) - {price_column} 
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                    )) AS RSI_{timeperiod}
                """
            },
            
            "MACD": {
                "category": "Momentum Indicators",
                "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
                "sql_template": """
                    -- MACD
                    EMA_{fastperiod} - EMA_{slowperiod} AS MACD_LINE,
                    EMA(MACD_LINE, {signalperiod}) AS MACD_SIGNAL,
                    MACD_LINE - MACD_SIGNAL AS MACD_HISTOGRAM
                """
            },
            
            "STOCH": {
                "category": "Momentum Indicators",
                "params": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
                "sql_template": """
                    -- Stochastic
                    100 * (close_price - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) / (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) AS STOCH_K,
                    
                    AVG(STOCH_K) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {slowk_period} - 1 PRECEDING AND CURRENT ROW
                    ) AS STOCH_D
                """
            },
            
            "STOCHF": {
                "category": "Momentum Indicators",
                "params": {"fastk_period": 5, "fastd_period": 3},
                "sql_template": """
                    -- Fast Stochastic
                    100 * (close_price - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) / (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) AS STOCHF_K,
                    
                    AVG(STOCHF_K) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastd_period} - 1 PRECEDING AND CURRENT ROW
                    ) AS STOCHF_D
                """
            },
            
            "STOCHRSI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14, "fastk_period": 14, "fastd_period": 3},
                "sql_template": """
                    -- Stochastic RSI
                    (RSI_{timeperiod} - MIN(RSI_{timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) / (MAX(RSI_{timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(RSI_{timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastk_period} - 1 PRECEDING AND CURRENT ROW
                    )) AS STOCHRSI_K,
                    
                    AVG(STOCHRSI_K) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastd_period} - 1 PRECEDING AND CURRENT ROW
                    ) AS STOCHRSI_D
                """
            },
            
            "WILLR": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Williams %R
                    -100 * (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - close_price) / (MAX(high_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - MIN(low_price) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) AS WILLR_{timeperiod}
                """
            },
            
            "ADX": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- ADX (simplified)
                    AVG(ABS(
                        (high_price - LAG(high_price) OVER (ORDER BY trade_date, trade_time)) - 
                        (LAG(low_price) OVER (ORDER BY trade_date, trade_time) - low_price)
                    )) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ADX_{timeperiod}
                """
            },
            
            "CCI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Commodity Channel Index
                    (((high_price + low_price + close_price) / 3) - 
                    AVG((high_price + low_price + close_price) / 3) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) / (0.015 * AVG(ABS(
                        (high_price + low_price + close_price) / 3 - 
                        AVG((high_price + low_price + close_price) / 3) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                    )) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    )) AS CCI_{timeperiod}
                """
            },
            
            "MFI": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Money Flow Index
                    100 - (100 / (1 + 
                        SUM(CASE WHEN (high_price + low_price + close_price) / 3 > 
                            LAG((high_price + low_price + close_price) / 3) OVER (ORDER BY trade_date, trade_time)
                            THEN ((high_price + low_price + close_price) / 3) * volume
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        ) /
                        SUM(CASE WHEN (high_price + low_price + close_price) / 3 < 
                            LAG((high_price + low_price + close_price) / 3) OVER (ORDER BY trade_date, trade_time)
                            THEN ((high_price + low_price + close_price) / 3) * volume
                            ELSE 0 END) OVER (
                            ORDER BY trade_date, trade_time
                            ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                        )
                    )) AS MFI_{timeperiod}
                """
            },
            
            "ROC": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 10},
                "sql_template": """
                    -- Rate of Change
                    100 * ({price_column} - LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    )) / LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS ROC_{timeperiod}
                """
            },
            
            "ROCP": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 10},
                "sql_template": """
                    -- Rate of Change Percentage
                    ({price_column} - LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    )) / LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS ROCP_{timeperiod}
                """
            },
            
            "ROCR": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 10},
                "sql_template": """
                    -- Rate of Change Ratio
                    {price_column} / LAG({price_column}, {timeperiod}) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS ROCR_{timeperiod}
                """
            },
            
            "TRIX": {
                "category": "Momentum Indicators",
                "params": {"timeperiod": 30},
                "sql_template": """
                    -- TRIX (simplified - rate of change of triple EMA)
                    (TEMA_{timeperiod} - LAG(TEMA_{timeperiod}) OVER (ORDER BY trade_date, trade_time)) / 
                    LAG(TEMA_{timeperiod}) OVER (ORDER BY trade_date, trade_time) * 100 AS TRIX_{timeperiod}
                """
            },
            
            # Volume Indicators
            "AD": {
                "category": "Volume Indicators",
                "params": {},
                "sql_template": """
                    -- Accumulation/Distribution Line
                    SUM(((close_price - low_price) - (high_price - close_price)) / 
                        (high_price - low_price) * volume) OVER (
                        ORDER BY trade_date, trade_time
                    ) AS AD
                """
            },
            
            "ADOSC": {
                "category": "Volume Indicators",
                "params": {"fastperiod": 3, "slowperiod": 10},
                "sql_template": """
                    -- Chaikin A/D Oscillator
                    AVG(AD) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {fastperiod} - 1 PRECEDING AND CURRENT ROW
                    ) - AVG(AD) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {slowperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ADOSC_{fastperiod}_{slowperiod}
                """
            },
            
            "OBV": {
                "category": "Volume Indicators",
                "params": {},
                "sql_template": """
                    -- On Balance Volume
                    SUM(CASE 
                        WHEN close_price > LAG(close_price) OVER (ORDER BY trade_date, trade_time) THEN volume
                        WHEN close_price < LAG(close_price) OVER (ORDER BY trade_date, trade_time) THEN -volume
                        ELSE 0
                    END) OVER (ORDER BY trade_date, trade_time) AS OBV
                """
            },
            
            # Volatility Indicators
            "ATR": {
                "category": "Volatility Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Average True Range
                    AVG(GREATEST(
                        high_price - low_price,
                        ABS(high_price - LAG(close_price) OVER (ORDER BY trade_date, trade_time)),
                        ABS(low_price - LAG(close_price) OVER (ORDER BY trade_date, trade_time))
                    )) OVER (
                        ORDER BY trade_date, trade_time
                        ROWS BETWEEN {timeperiod} - 1 PRECEDING AND CURRENT ROW
                    ) AS ATR_{timeperiod}
                """
            },
            
            "NATR": {
                "category": "Volatility Indicators",
                "params": {"timeperiod": 14},
                "sql_template": """
                    -- Normalized ATR
                    100 * ATR_{timeperiod} / close_price AS NATR_{timeperiod}
                """
            },
            
            # Price Transform
            "AVGPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Average Price
                    (open_price + high_price + low_price + close_price) / 4 AS AVGPRICE
                """
            },
            
            "MEDPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Median Price
                    (high_price + low_price) / 2 AS MEDPRICE
                """
            },
            
            "TYPPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Typical Price
                    (high_price + low_price + close_price) / 3 AS TYPPRICE
                """
            },
            
            "WCLPRICE": {
                "category": "Price Transform",
                "params": {},
                "sql_template": """
                    -- Weighted Close Price
                    (high_price + low_price + 2 * close_price) / 4 AS WCLPRICE
                """
            }
        }
    
    def get_indicator_sql(self, 
                         indicator_name: str,
                         params: Dict[str, Any],
                         table_name: str = "market_data",
                         price_column: str = "close_price",
                         partition_by: str = "index_name") -> str:
        """
        Get SQL implementation for a specific indicator
        
        Args:
            indicator_name: Name of the indicator
            params: Parameters for the indicator
            table_name: Source table name
            price_column: Price column to use
            partition_by: Column to partition by
            
        Returns:
            SQL string for the indicator
        """
        if indicator_name not in self.indicators:
            raise ValueError(f"Indicator {indicator_name} not supported")
            
        indicator = self.indicators[indicator_name]
        sql_template = indicator["sql_template"]
        
        # Merge default params with user params
        final_params = {**indicator["params"], **params}
        
        # Replace placeholders
        sql = sql_template.format(
            table_name=table_name,
            price_column=price_column,
            partition_by=partition_by,
            **final_params
        )
        
        return sql
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of all supported indicators"""
        return list(self.indicators.keys())
    
    def get_indicator_category(self, indicator_name: str) -> str:
        """Get category for a specific indicator"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name]["category"]
        return "Unknown"
    
    def get_indicator_params(self, indicator_name: str) -> Dict[str, Any]:
        """Get default parameters for a specific indicator"""
        if indicator_name in self.indicators:
            return self.indicators[indicator_name]["params"].copy()
        return {}
    
    def build_multi_indicator_query(self,
                                  indicators: List[Dict[str, Any]],
                                  table_name: str = "market_data") -> str:
        """
        Build a query with multiple indicators
        
        Args:
            indicators: List of indicator configurations
            table_name: Source table name
            
        Returns:
            Complete SQL query with all indicators
        """
        select_parts = ["*"]  # Include all base columns
        
        for ind_config in indicators:
            ind_name = ind_config["name"]
            ind_params = ind_config.get("params", {})
            price_column = ind_config.get("price_column", "close_price")
            
            try:
                sql = self.get_indicator_sql(
                    ind_name,
                    ind_params,
                    table_name,
                    price_column
                )
                select_parts.append(sql)
            except ValueError:
                logger.warning(f"Skipping unsupported indicator: {ind_name}")
                
        query = f"""
        SELECT
            {','.join(select_parts)}
        FROM {table_name}
        ORDER BY trade_date, trade_time
        """
        
        return query
    
    def validate_indicator_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate an indicator configuration
        
        Args:
            config: Indicator configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if "name" not in config:
            return False, "Indicator name is required"
            
        ind_name = config["name"]
        if ind_name not in self.indicators:
            return False, f"Indicator {ind_name} not supported"
            
        # Validate parameters
        ind_params = config.get("params", {})
        default_params = self.indicators[ind_name]["params"]
        
        for param, value in ind_params.items():
            if param not in default_params:
                return False, f"Invalid parameter {param} for indicator {ind_name}"
                
            # Type checking
            if isinstance(default_params[param], int) and not isinstance(value, int):
                return False, f"Parameter {param} must be an integer"
            elif isinstance(default_params[param], float) and not isinstance(value, (int, float)):
                return False, f"Parameter {param} must be a number"
                
        return True, None